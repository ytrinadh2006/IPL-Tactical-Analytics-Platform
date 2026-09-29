import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

function NumberValue({ value, decimals = 0 }) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "—";
  }

  return decimals > 0 ? n.toFixed(decimals) : n.toLocaleString();
}

function App() {
  const [summary, setSummary] = useState({});
  const [teams, setTeams] = useState([]);
  const [players, setPlayers] = useState([]);
  const [venues, setVenues] = useState([]);

  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [cmp, setCmp] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(API + "/summary").then((r) => r.json()),
      fetch(API + "/teams").then((r) => r.json()),
      fetch(API + "/players?limit=10").then((r) => r.json()),
      fetch(API + "/venues").then((r) => r.json()),
    ])
      .then(([s, t, p, v]) => {
        setSummary(s || {});

        // Teams API can return:
        // { teams: ["Mumbai Indians", ...] }
        // or directly ["Mumbai Indians", ...]
        const teamList = Array.isArray(t)
          ? t
          : Array.isArray(t?.teams)
          ? t.teams
          : Array.isArray(t?.data)
          ? t.data
          : [];

        // Players API can return:
        // { players: [...] }
        // or directly [...]
        const playerList = Array.isArray(p)
          ? p
          : Array.isArray(p?.players)
          ? p.players
          : Array.isArray(p?.data)
          ? p.data
          : [];

        // Venues API can return:
        // { venues: [...] }
        // or directly [...]
        const venueList = Array.isArray(v)
          ? v
          : Array.isArray(v?.venues)
          ? v.venues
          : Array.isArray(v?.data)
          ? v.data
          : [];

        setTeams(teamList);
        setPlayers(playerList);
        setVenues(venueList);
      })
      .catch((err) => {
        console.error(err);
        setError("Start the FastAPI server to load live data.");
      });
  }, []);

  const getTeamName = (team) => {
    if (typeof team === "string") return team;

    return (
      team?.team ||
      team?.team_name ||
      team?.name ||
      ""
    );
  };

  const getPlayerName = (player) => {
    if (typeof player === "string") return player;

    return (
      player?.player ||
      player?.player_name ||
      player?.name ||
      ""
    );
  };

  const getPlayerImpact = (player) => {
    if (typeof player === "object" && player !== null) {
      return (
        player?.impact_score ??
        player?.impact ??
        player?.score ??
        null
      );
    }

    return null;
  };

  const getVenueName = (venue) => {
    if (typeof venue === "string") return venue;

    return (
      venue?.venue ||
      venue?.venue_name ||
      venue?.name ||
      ""
    );
  };

  const getVenueRuns = (venue) => {
    if (typeof venue === "object" && venue !== null) {
      return (
        venue?.avg_runs_per_match ??
        venue?.average_runs_per_match ??
        venue?.avg_first_innings_score ??
        venue?.average_first_innings_score ??
        null
      );
    }

    return null;
  };

  const compare = () => {
    if (!a || !b) {
      return;
    }

    fetch(
      API +
        "/teams/compare/" +
        encodeURIComponent(a) +
        "/" +
        encodeURIComponent(b)
    )
      .then((r) => r.json())
      .then((data) => {
        setCmp(data);
        setError("");
      })
      .catch((err) => {
        console.error(err);
        setError("Unable to compare these teams.");
      });
  };

  const comparisonTeams = cmp
    ? [
        cmp?.data?.team_a,
        cmp?.data?.team_b,
      ].filter(Boolean)
    : [];

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">IPL ANALYTICS</p>

          <h1>IPL Intelligence</h1>

          <p className="sub">
            Historical cricket data turned into practical player, team and
            venue insights.
          </p>
        </div>

        <div className="pill">
          Data + Analytics + ML
        </div>
      </header>

      {/* SUMMARY */}
      <section className="grid stats">
        {[
          ["Matches", summary.matches],
          ["Deliveries", summary.deliveries],
          ["Seasons", summary.seasons],
          ["Teams", summary.teams],
          ["Venues", summary.venues],
          ["Players", summary.players],
        ].map((x) => (
          <div className="card" key={x[0]}>
            <span>{x[0]}</span>

            <strong>
              <NumberValue value={x[1]} />
            </strong>
          </div>
        ))}
      </section>

      {error && (
        <div className="notice">
          {error}
        </div>
      )}

      {/* TEAM COMPARISON */}
      <section className="panel">
        <div className="section-head">
          <div>
            <h2>Team comparison</h2>

            <p>
              Compare overall historical results from the processed IPL
              dataset.
            </p>
          </div>
        </div>

        <div className="controls">
          <select
            value={a}
            onChange={(e) => setA(e.target.value)}
          >
            <option value="">Team 1</option>

            {teams.map((team, index) => {
              const name = getTeamName(team);

              return (
                <option
                  value={name}
                  key={name || index}
                >
                  {name}
                </option>
              );
            })}
          </select>

          <select
            value={b}
            onChange={(e) => setB(e.target.value)}
          >
            <option value="">Team 2</option>

            {teams.map((team, index) => {
              const name = getTeamName(team);

              return (
                <option
                  value={name}
                  key={name || index}
                >
                  {name}
                </option>
              );
            })}
          </select>

          <button
            onClick={compare}
            disabled={!a || !b}
          >
            Compare
          </button>
        </div>

        {comparisonTeams.length > 0 && (
          <div className="compare">
            {comparisonTeams.map((team, index) => {
              const teamName =
                typeof team === "string"
                  ? team
                  : team?.team ||
                    team?.team_name ||
                    team?.name ||
                    `Team ${index + 1}`;

              return (
                <div
                  className="compare-card"
                  key={teamName}
                >
                  <h3>{teamName}</h3>

                  <div className="big">
                    <NumberValue
                      value={
                        team?.win_pct ??
                        team?.win_percentage ??
                        team?.wins_percentage
                      }
                      decimals={1}
                    />
                    %
                  </div>

                  <span>Win rate</span>

                  <hr />

                  <p>
                    Matches{" "}
                    <b>
                      <NumberValue value={team?.matches} />
                    </b>
                  </p>

                  <p>
                    Wins{" "}
                    <b>
                      <NumberValue value={team?.wins} />
                    </b>
                  </p>

                  <p>
                    Losses{" "}
                    <b>
                      <NumberValue value={team?.losses} />
                    </b>
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* PLAYER IMPACT + VENUE */}
      <section className="two">
        <div className="panel">
          <h2>Top player impact</h2>

          <p>
            Composite ranking based on normalized batting, bowling and
            available fielding contributions.
          </p>

          <div className="rows">
            {players.length === 0 ? (
              <div className="row">
                <span>No player data available</span>
                <b>—</b>
              </div>
            ) : (
              players.slice(0, 10).map((player, index) => {
                const name = getPlayerName(player);
                const impact = getPlayerImpact(player);

                return (
                  <div
                    className="row"
                    key={name || index}
                  >
                    <span>
                      {index + 1}. {name || "Unknown player"}
                    </span>

                    <b>
                      {impact === null ? (
                        "—"
                      ) : (
                        <NumberValue
                          value={impact}
                          decimals={1}
                        />
                      )}
                    </b>
                  </div>
                );
              })
            )}
          </div>
        </div>

        <div className="panel">
          <h2>Venue overview</h2>

          <p>
            Historical scoring and chasing behavior.
          </p>

          <div className="rows">
            {venues.length === 0 ? (
              <div className="row">
                <span>No venue data available</span>
                <b>—</b>
              </div>
            ) : (
              venues.slice(0, 8).map((venue, index) => {
                const name = getVenueName(venue);
                const runs = getVenueRuns(venue);

                return (
                  <div
                    className="row"
                    key={name || index}
                  >
                    <span>
                      {name || `Venue ${index + 1}`}
                    </span>

                    <b>
                      {runs === null
                        ? "—"
                        : Number(runs).toFixed(1)}
                    </b>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </section>

      {/* ANALYSIS SNAPSHOTS */}
      <section className="panel">
        <h2>Analysis snapshots</h2>

        <p>
          Static plots are generated from the same processed dataset used by
          the analytical scripts.
        </p>

        <div className="plots">
          {[
            ["/team_wins.png", "Team wins"],
            ["/season_matches.png", "Matches by season"],
            ["/top_run_scorers.png", "Top run scorers"],
            ["/top_wicket_takers.png", "Top wicket takers"],
            ["/player_impact.png", "Player impact"],
          ].map((x) => (
            <figure key={x[0]}>
              <img
                src={x[0]}
                alt={x[1]}
              />

              <figcaption>
                {x[1]}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <footer>
        Built as an IPL historical analytics project.
        Predictions are estimates, not guarantees.
      </footer>
    </main>
  );
}

createRoot(
  document.getElementById("root")
).render(
  <App />
);