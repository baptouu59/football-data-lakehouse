import plotly.express as px
import streamlit as st
from pyspark.sql import SparkSession

# ==============================================================================
# CONFIGURATION
# ==============================================================================

st.set_page_config(page_title="Football Data Lakehouse", page_icon="⚽", layout="wide")

st.title("⚽ Football Data Lakehouse")
st.markdown(
    """
    Analyse des performances des clubs de football à partir
    des tables Gold du Data Lakehouse.
    """
)


# ==============================================================================
# SPARK
# ==============================================================================


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder.appName("FootballDashboard")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@st.cache_data(ttl=60)
def load_gold_table(path):
    spark = get_spark()

    return spark.read.format("delta").load(path).toPandas()


# ==============================================================================
# CHARGEMENT DES TABLES GOLD
# ==============================================================================

club_goals = load_gold_table("/lakehouse/gold/club_goals_pareto")

yearly_goals = load_gold_table("/lakehouse/gold/goals_evolution_yearly")

top_clubs = load_gold_table("/lakehouse/gold/top_club_by_division_year")

club_segments = load_gold_table("/lakehouse/gold/club_segments")


# ==============================================================================
# KPI
# ==============================================================================

st.subheader("Vue générale")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Clubs analysés", f"{club_goals['team'].nunique():,}")

col2.metric("Matchs représentés", f"{int(yearly_goals['nb_matches'].sum()):,}")

col3.metric("Buts analysés", f"{int(yearly_goals['total_goals'].sum()):,}")

col4.metric("Clubs segmentés", f"{len(club_segments):,}")


st.divider()


# ==============================================================================
# 1. EVOLUTION TEMPORELLE DES BUTS
# ==============================================================================

st.subheader("1. Évolution du nombre moyen de buts par match")

fig_year = px.line(
    yearly_goals,
    x="year",
    y="avg_goals_per_match",
    markers=True,
    labels={"year": "Année", "avg_goals_per_match": "Buts moyens par match"},
)

st.plotly_chart(fig_year, use_container_width=True)


with st.expander("Requête SQL & justification statistique"):
    st.code(
        """
WITH yearly_stats AS (
    SELECT
        YEAR(matchdate) AS year,
        COUNT(*) AS nb_matches,
        AVG(fthome + ftaway) AS avg_goals_per_match,
        SUM(fthome + ftaway) AS total_goals
    FROM v_silver
    GROUP BY YEAR(matchdate)
)

SELECT *,
       LAG(avg_goals_per_match, 1)
       OVER (ORDER BY year) AS previous_year_avg
FROM yearly_stats
ORDER BY year;
        """,
        language="sql",
    )

    st.markdown(
        """
        La moyenne de buts par match permet de comparer les années
        indépendamment du nombre de matchs disponibles.

        La fonction `LAG()` permet de comparer chaque année avec
        l'année précédente et d'étudier l'évolution temporelle.
        """
    )


st.divider()


# ==============================================================================
# 2. CLASSEMENT DES CLUBS PAR NOMBRE DE BUTS
# ==============================================================================

st.subheader("2. Clubs ayant marqué le plus de buts")

top_n = st.slider("Nombre de clubs à afficher", min_value=5, max_value=30, value=15)

top_goals = club_goals.sort_values("total_goals", ascending=False).head(top_n)

fig_goals = px.bar(
    top_goals,
    x="team",
    y="total_goals",
    hover_data=["nb_matchs", "avg_goals_per_match", "avg_elo"],
    labels={"team": "Club", "total_goals": "Nombre total de buts"},
)

st.plotly_chart(fig_goals, use_container_width=True)


with st.expander("Requête SQL & justification statistique"):
    st.code(
        """
WITH club_stats AS (
    SELECT
        team,
        COUNT(*) AS nb_matchs,
        SUM(goals_scored) AS total_goals,
        AVG(goals_scored) AS avg_goals_per_match,
        PERCENTILE_APPROX(goals_scored, 0.50)
            AS median_goals
    FROM team_matches
    GROUP BY team
)

SELECT *,
       DENSE_RANK()
       OVER (ORDER BY total_goals DESC)
           AS goals_rank
FROM club_stats;
        """,
        language="sql",
    )

    st.markdown(
        """
        Le classement est basé sur le nombre total de buts marqués.

        La médiane et l'IQR ont également été calculés dans la table
        Gold afin de disposer d'indicateurs plus robustes face aux
        valeurs extrêmes.
        """
    )


st.divider()


# ==============================================================================
# 3. SEGMENTATION K-MEANS
# ==============================================================================

st.subheader("3. Segmentation des clubs")

segments = sorted(club_segments["segment"].dropna().unique())

selected_segments = st.multiselect("Segments à afficher", options=segments, default=segments)

filtered_segments = club_segments[club_segments["segment"].isin(selected_segments)]

fig_clusters = px.scatter(
    filtered_segments,
    x="avg_elo",
    y="win_rate",
    color="segment",
    size="nb_matches",
    hover_name="team",
    hover_data=["avg_goals_scored", "avg_goals_conceded"],
    labels={"avg_elo": "Elo moyen", "win_rate": "Taux de victoire", "segment": "Segment"},
)

st.plotly_chart(fig_clusters, use_container_width=True)


with st.expander("Méthode & justification statistique"):
    st.markdown(
        """
        Les clubs ont été segmentés avec un algorithme **K-Means**.

        Variables utilisées :

        - nombre de matchs ;
        - Elo moyen ;
        - buts marqués moyens ;
        - buts encaissés moyens ;
        - taux de victoire.

        `log1p()` a été appliqué au nombre de matchs afin de réduire
        son asymétrie.

        Les variables ont ensuite été standardisées avec
        `StandardScaler`.

        Le nombre de clusters a été choisi avec le coefficient de
        silhouette :

        - k = 2 : 0,315
        - **k = 3 : 0,324**
        - k = 4 : 0,245
        - k = 5 : 0,256
        - k = 6 : 0,256

        Le meilleur score est obtenu avec **3 clusters**.
        """
    )


st.divider()


# ==============================================================================
# 4. MEILLEUR CLUB PAR CHAMPIONNAT
# ==============================================================================

st.subheader("4. Meilleur club par championnat")

available_years = sorted(top_clubs["year"].dropna().astype(int).unique(), reverse=True)

# 2025 par défaut car 2026 est une année partielle
default_year = 2025 if 2025 in available_years else available_years[0]

selected_year = st.selectbox(
    "Choisir une année", options=available_years, index=available_years.index(default_year)
)

# Message si l'utilisateur sélectionne 2026
if selected_year == 2026:
    st.info(
        "Les données 2026 sont partielles : "
        "les classements affichés ne représentent pas forcément une saison complète."
    )

year_data = top_clubs[top_clubs["year"] == selected_year].copy()

year_data = year_data.sort_values("points", ascending=False)

# Nom lisible directement sur le graphique
year_data["club_championship"] = year_data["team"] + " (" + year_data["division"] + ")"

fig_top = px.bar(
    year_data,
    x="points",
    y="club_championship",
    orientation="h",
    hover_name="team",
    hover_data={
        "division": True,
        "wins": True,
        "draws": True,
        "losses": True,
        "goal_difference": True,
        "club_championship": False,
    },
    labels={
        "club_championship": "Club / championnat",
        "points": "Points",
        "division": "Championnat",
        "wins": "Victoires",
        "draws": "Nuls",
        "losses": "Défaites",
        "goal_difference": "Différence de buts",
    },
)

fig_top.update_layout(height=max(600, len(year_data) * 32), yaxis={"categoryorder": "total ascending"})

st.plotly_chart(fig_top, width="stretch")

with st.expander("Requête SQL & justification métier"):
    st.code(
        """
ROW_NUMBER() OVER (
    PARTITION BY year, division
    ORDER BY
        points DESC,
        goal_difference DESC,
        goals_for DESC
) AS ranking
        """,
        language="sql",
    )

    st.markdown(
        """
        Pour chaque championnat et chaque année, les clubs sont
        classés selon leur nombre de points.

        En cas d'égalité, la différence de buts puis le nombre
        de buts marqués permettent de départager les équipes.

        `ROW_NUMBER()` permet ensuite de conserver le club classé
        numéro 1 de chaque championnat.
        """
    )


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()

st.caption("Data Lakehouse Football — HDFS • Delta Lake • Spark SQL • SciPy • K-Means • Streamlit")
