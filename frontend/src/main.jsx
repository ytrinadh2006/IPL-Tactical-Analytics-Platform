import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

function NumberValue({ value, decimals = 0 }) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return decimals > 0 ? n.toFixed(decimals) : n.toLocaleString();
}

function App() {
  const [summary, setSummary] = useState({});
  const [teams, setTeams] = useState([]);
  const [players, setPlayers] = useState([]);
  const [batters, setBatters] = useState([]);
  const [bowlers, setBowlers] = useState([]);
  const [fielders, setFielders] = useState([]);
  const [venues, setVenues] = useState([]);

  // Team comparison
  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");
  const [cmp, setCmp] = useState(null);

  // ML Prediction
  const [predTeamA, setPredTeamA] = useState("");
  const [predTeamB, setPredTeamB] = useState("");
  const [predVenue, setPredVenue] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);

  // Player tabs
  const [playerTab, setPlayerTab] = useState("impact"); // impact, batting, bowling, fielding
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(API + "/summary").then((r) => r.json()).catch(() => ({})),
      fetch(API + "/teams").then((r) => r.json()).catch(() => []),
      fetch(API + "/players?limit=10").then((r) => r.json()).catch(() => []),
      fetch(API + "/venues").then((r) => r.json()).catch(() => []),
      fetch(API + "/batting?limit=10").then((r) => r.json()).catch(() => []),
      fetch(API + "/bowling?limit=10").then((r) => r.json()).catch(() => []),
      fetch(API + "/fielding?limit=10").then((r) => r.json()).catch(() => []),
    ])
      .then(([s, t, p, v, bat, bowl, fld]) => {
        setSummary(s || {});

        const teamList = Array.isArray(t)
          ? t
          : Array.isArray(t?.teams)
          ? t.teams
          : Array.isArray(t?.data)
          ? t.data
          : [];

        const playerList = Array.isArray(p)
          ? p
          : Array.isArray(p?.players)
          ? p.players
          : Array.isArray(p?.data)
          ? p.data
          : [];

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
        if (Array.isArray(bat)) setBatters(bat);
        if (Array.isArray(bowl)) setBowlers(bowl);
        if (Array.isArray(fld)) setFielders(fld);

        if (teamList.length >= 2) {
          const t1 = getTeamName(teamList[0]);
          const t2 = getTeamName(teamList[1]);
          setTeamA(t1);
          setTeamB(t2);
          setPredTeamA(t1);
          setPredTeamB(t2);
        }
      })
      .catch((err) => {
        console.error(err);
        setError("Start the FastAPI server (uvicorn backend.app.main:app) to load live data.");
      });
  }, []);

  const getTeamName = (t) => {
    if (typeof t === "string") return t;
    return t?.team || t?.team_name || t?.name || "";
  };

  const getPlayerName = (p) => {
    if (typeof p === "string") return p;
    return p?.player || p?.player_name || p?.name || "";
  };

  const getPlayerImpact = (p) => {
    if (typeof p === "object" && p !== null) {
      return p?.impact_score ?? p?.impact ?? p?.score ?? null;
    }
    return null;
  };

  const getVenueName = (v) => {
    if (typeof v === "string") return v;
    return v?.venue || v?.venue_name || v?.name || "";
  };

  const compare = () => {
    if (!teamA || !teamB) return;

    fetch(`${API}/teams/compare/${encodeURIComponent(teamA)}/${encodeURIComponent(teamB)}`)
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

  const runPrediction = () => {
    if (!predTeamA || !predTeamB) return;
    setPredicting(true);
    setPrediction(null);

    fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team_a: predTeamA,
        team_b: predTeamB,
        venue: predVenue || undefined,
      }),
    })
      .then((r) => r.json())
      .then((data) => {
        setPredicting(false);
        const p = data?.prediction || data;
        setPrediction(p);
      })
      .catch((err) => {
        setPredicting(false);
        console.error(err);
        setError("Prediction service request failed.");
      });
  };

  const comparisonDataA = cmp?.team_a || cmp?.data?.team_a;
  const comparisonDataB = cmp?.team_b || cmp?.data?.team_b;

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">IPL Historical Intelligence Platform (2008 – 2026)</p>
          <h1>IPL Tactical Analytics</h1>
          <p className="sub">
            Comprehensive historical intelligence covering 2008 through 2026 seasons.
            Batting, bowling, fielding, head-to-head match dynamics, and machine learning outcome estimation.
          </p>
        </div>
        <div className="pill">
          2008 – 2026 Data Engine
        </div>
      </header>

      {/* SUMMARY */}
      <section className="grid stats">
        {[
          ["Matches", summary.matches || 1243],
          ["Deliveries", summary.deliveries || 295732],
          ["Seasons", summary.seasons || 19],
          ["Teams", summary.teams || 15],
          ["Venues", summary.venues || 60],
          ["Players", summary.players || summary.players_batting || 800],
        ].map(([label, val]) => (
          <div className="card" key={label}>
            <span>{label}</span>
            <strong>
              <NumberValue value={val} />
            </strong>
          </div>
        ))}
      </section>

      {error && <div className="notice">{error}</div>}

      {/* ML PREDICTION SECTION */}
      <section className="panel">
        <div className="section-head">
          <div>
            <h2>ML Match Outcome Predictor</h2>
            <p>
              Pre-match outcome estimation using historical form, team ratings, and venue characteristics.
            </p>
          </div>
        </div>

        <div className="controls">
          <select value={predTeamA} onChange={(e) => setPredTeamA(e.target.value)}>
            <option value="">Select Team A</option>
            {teams.map((t, idx) => {
              const name = getTeamName(t);
              return <option value={name} key={name || idx}>{name}</option>;
            })}
          </select>

          <span style={{ color: "#8b949e", fontWeight: "bold" }}>VS</span>

          <select value={predTeamB} onChange={(e) => setPredTeamB(e.target.value)}>
            <option value="">Select Team B</option>
            {teams.map((t, idx) => {
              const name = getTeamName(t);
              return <option value={name} key={name || idx}>{name}</option>;
            })}
          </select>

          <select value={predVenue} onChange={(e) => setPredVenue(e.target.value)}>
            <option value="">Any Stadium / Neutral</option>
            {venues.map((v, idx) => {
              const name = getVenueName(v);
              return <option value={name} key={name || idx}>{name}</option>;
            })}
          </select>

          <button onClick={runPrediction} disabled={predicting || !predTeamA || !predTeamB}>
            {predicting ? "Predicting..." : "Predict Winner"}
          </button>
        </div>

        {prediction && (
          <div className="pred-result">
            {(() => {
              const probA =
                prediction.team_a_probability !== undefined
                  ? prediction.team_a_probability * 100
                  : prediction.team1_win_probability !== undefined
                  ? prediction.team1_win_probability
                  : 50;
              const probB =
                prediction.team_b_probability !== undefined
                  ? prediction.team_b_probability * 100
                  : prediction.team2_win_probability !== undefined
                  ? prediction.team2_win_probability
                  : 50;
              const nameA = prediction.team_a || predTeamA;
              const nameB = prediction.team_b || predTeamB;
              const favored = probA >= probB ? nameA : nameB;

              return (
                <div>
                  <div className="pred-header">
                    <div>
                      <span style={{ color: "#8b949e", fontSize: "12px", textTransform: "uppercase" }}>
                        Estimated Advantage
                      </span>
                      <div className="pred-winner">{favored} Favored</div>
                    </div>
                    <div style={{ color: "#8b949e", fontSize: "13px" }}>
                      Model: {prediction.model || "Logistic Regression V2"}
                    </div>
                  </div>

                  <div className="progress-container">
                    <div className="progress-bar-a" style={{ width: `${probA}%` }} />
                    <div className="progress-bar-b" style={{ width: `${probB}%` }} />
                  </div>

                  <div className="progress-labels">
                    <span style={{ color: "#58a6ff" }}>{nameA}: {probA.toFixed(1)}%</span>
                    <span style={{ color: "#d29922" }}>{nameB}: {probB.toFixed(1)}%</span>
                  </div>

                  {prediction.reference_match_date && (
                    <p style={{ margin: "14px 0 0", color: "#8b949e", fontSize: "12px" }}>
                      Grounding: Most recent head-to-head encounter was on {prediction.reference_match_date} at {prediction.venue || "venue"}.
                    </p>
                  )}
                </div>
              );
            })()}
          </div>
        )}
      </section>

      {/* TEAM COMPARISON */}
      <section className="panel">
        <div className="section-head">
          <div>
            <h2>Team Comparison & Head-to-Head</h2>
            <p>Compare all-time IPL franchise performance across 2008 – 2026 seasons.</p>
          </div>
        </div>

        <div className="controls">
          <select value={teamA} onChange={(e) => setTeamA(e.target.value)}>
            <option value="">Select Team 1</option>
            {teams.map((t, idx) => {
              const name = getTeamName(t);
              return <option value={name} key={name || idx}>{name}</option>;
            })}
          </select>

          <select value={teamB} onChange={(e) => setTeamB(e.target.value)}>
            <option value="">Select Team 2</option>
            {teams.map((t, idx) => {
              const name = getTeamName(t);
              return <option value={name} key={name || idx}>{name}</option>;
            })}
          </select>

          <button onClick={compare} disabled={!teamA || !teamB}>
            Compare Teams
          </button>
        </div>

        {(comparisonDataA || comparisonDataB) && (
          <div className="compare">
            {[comparisonDataA, comparisonDataB].filter(Boolean).map((tData, idx) => {
              const profile = tData.overall || tData;
              const batting = tData.batting || {};
              const bowling = tData.bowling || {};
              const name = profile.team || (idx === 0 ? teamA : teamB);

              return (
                <div className="compare-card" key={name}>
                  <h3>{name}</h3>
                  <div className="big">
                    <NumberValue value={profile.win_pct || profile.win_percentage} decimals={1} />%
                  </div>
                  <span>Win Rate (2008–2026)</span>
                  <hr />
                  <p>
                    Matches Played <b><NumberValue value={profile.matches} /></b>
                  </p>
                  <p>
                    Total Wins <b><NumberValue value={profile.wins} /></b>
                  </p>
                  <p>
                    Total Losses <b><NumberValue value={profile.losses} /></b>
                  </p>
                  {batting.strike_rate && (
                    <p>
                      Batting Strike Rate <b><NumberValue value={batting.strike_rate} decimals={1} /></b>
                    </p>
                  )}
                  {bowling.economy && (
                    <p>
                      Bowling Economy <b><NumberValue value={bowling.economy} decimals={2} /></b>
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* PLAYER & VENUE ANALYTICS */}
      <section className="two">
        {/* PLAYER ANALYTICS TABS */}
        <div className="panel">
          <h2>Player Performance (2008 – 2026)</h2>
          <p>Explore batting, bowling, fielding, and overall impact rankings.</p>

          <div className="tabs">
            <button
              className={`tab-btn ${playerTab === "impact" ? "active" : ""}`}
              onClick={() => setPlayerTab("impact")}
            >
              Player Impact
            </button>
            <button
              className={`tab-btn ${playerTab === "batting" ? "active" : ""}`}
              onClick={() => setPlayerTab("batting")}
            >
              Batting
            </button>
            <button
              className={`tab-btn ${playerTab === "bowling" ? "active" : ""}`}
              onClick={() => setPlayerTab("bowling")}
            >
              Bowling
            </button>
            <button
              className={`tab-btn ${playerTab === "fielding" ? "active" : ""}`}
              onClick={() => setPlayerTab("fielding")}
            >
              Fielding
            </button>
          </div>

          <div className="rows">
            {playerTab === "impact" &&
              players.slice(0, 10).map((p, idx) => (
                <div className="row" key={getPlayerName(p) || idx}>
                  <span>{idx + 1}. {getPlayerName(p)}</span>
                  <b><NumberValue value={getPlayerImpact(p)} decimals={1} /> pts</b>
                </div>
              ))}

            {playerTab === "batting" &&
              batters.slice(0, 10).map((p, idx) => (
                <div className="row" key={p.player || idx}>
                  <span>{idx + 1}. {p.player}</span>
                  <b><NumberValue value={p.runs} /> runs ({Number(p.strike_rate || 0).toFixed(1)} SR)</b>
                </div>
              ))}

            {playerTab === "bowling" &&
              bowlers.slice(0, 10).map((p, idx) => (
                <div className="row" key={p.player || idx}>
                  <span>{idx + 1}. {p.player}</span>
                  <b><NumberValue value={p.wickets} /> wkts ({Number(p.economy || 0).toFixed(2)} econ)</b>
                </div>
              ))}

            {playerTab === "fielding" &&
              fielders.slice(0, 10).map((p, idx) => (
                <div className="row" key={p.player || idx}>
                  <span>{idx + 1}. {p.player}</span>
                  <b><NumberValue value={p.dismissals} /> dismissals ({p.catches || 0}c / {p.run_outs || 0}ro / {p.stumpings || 0}st)</b>
                </div>
              ))}
          </div>
        </div>

        {/* VENUE OVERVIEW */}
        <div className="panel">
          <h2>Venue Intelligence</h2>
          <p>Stadium behavior, first innings scoring, and chasing advantages.</p>

          <div className="rows">
            {venues.slice(0, 10).map((v, idx) => {
              const name = getVenueName(v);
              const avgScore = v?.avg_first_innings_score || v?.avg_runs_per_match;
              const chasePct = v?.chasing_win_pct;

              return (
                <div className="row" key={name || idx}>
                  <div style={{ maxWidth: "60%" }}>
                    <div style={{ fontWeight: 600 }}>{name}</div>
                    <span style={{ fontSize: "12px", color: "#8b949e" }}>{v.matches} matches</span>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <b>{avgScore ? Math.round(Number(avgScore)) : "—"} avg 1st inn</b>
                    <div style={{ fontSize: "12px", color: "#58a6ff" }}>
                      {chasePct ? `${Number(chasePct).toFixed(1)}% chase wins` : ""}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ANALYSIS SNAPSHOTS */}
      <section className="panel">
        <h2>Analysis Snapshots (2008 – 2026)</h2>
        <p>Visualizations regenerated from the 2008-2026 processed datasets.</p>

        <div className="plots">
          {[
            ["/team_wins.png", "Top IPL Teams by Total Wins (2008 - 2026)"],
            ["/season_matches.png", "IPL Matches by Season (2008 - 2026)"],
            ["/top_run_scorers.png", "All-Time Top Run Scorers (2008 - 2026)"],
            ["/top_wicket_takers.png", "All-Time Top Wicket Takers (2008 - 2026)"],
            ["/player_impact.png", "All-Time Player Impact Scores (2008 - 2026)"],
          ].map(([src, label]) => (
            <figure key={src}>
              <img src={src} alt={label} />
              <figcaption>{label}</figcaption>
            </figure>
          ))}
        </div>
      </section>

      <footer>
        IPL Intelligence & Tactical Analytics Platform • Up-to-date through 2026 Season • Trinadh Reddy
      </footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);