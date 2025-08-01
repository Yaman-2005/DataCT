from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import random
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'your-secret-key'

event_table_map = {
    '2023': {'lockin': 'lockin_23', 'stage1': 'stage1_23', 'tokyo': 'tokyo_23', 'champs': 'champs_23'},
    '2024': {'kickoff': 'kickoff_24', 'stage1': 'stage1_24', 'madrid': 'madrid_24', 'stage2': 'stage2_24', 'champs': 'champs_24'},
    '2025': {'kickoff': 'kickoff_25', 'stage1': 'stage1_25', 'bangkok': 'bangkok_25','toronto':'toronto_25'}
}
home_map = {
    'bangkok_25': '2025 Masters Bangkok',
    'toronto_25': '2025 Masters Toronto'
}
@app.route('/')
def home():
    top5 = []
    random_player = None
    try:
        conn = sqlite3.connect('/home/prlfive/DataCT/vct.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * from sqlite_master where type='table'")
        print(cursor.fetchall())
        # Get top 5 players by rating from toronto
        cursor.execute("SELECT player, team, rating, acs, adr FROM toronto_25 ORDER BY rating DESC LIMIT 5")
        top5 = cursor.fetchall()

        # Get random player from any of bangkok_25, tonronto_25 where rating > 1.0
        candidates = []
        for table in ['bangkok_25','toronto_25']:
            cursor.execute(f"SELECT player, team, rating, acs, adr, fkpr, fdpr,'{home_map[table]}' as event FROM {table} WHERE rating > 1.0")
            candidates.extend(cursor.fetchall())

        random_player = random.choice(candidates)
        if candidates:
            random_player = random.choice(candidates)

        conn.close()
    except Exception as e:
        print("Error fetching homepage metrics:", e)

    return render_template('home.html', top5=top5, random_player=random_player)


@app.route('/stats', methods=['GET', 'POST'])
def stats():
    error = None
    image_path = None
    stacked_df = pd.DataFrame()

    if 'stats_data' in session:
        try:
            stacked_df = pd.read_json(session['stats_data'])
            numeric_cols = ['RATING', 'ACS', 'K:D', 'KAST', 'ADR', 'FKPR', 'FDPR', 'APR', 'HS']
            for col in numeric_cols:
                if col in stacked_df.columns:
                    stacked_df[col] = stacked_df[col].astype(float).round(4)
        except:
            session.pop('stats_data', None)

    if request.method == 'POST':
        # DELETE row
        if 'delete_row' in request.form:
            delete_player = request.form.get('delete_player').strip()
            delete_event = request.form.get('delete_event').strip()
            mask = (stacked_df['PLAYER'].str.strip() == delete_player) & \
                   (stacked_df['Event'].str.strip() == delete_event)
            if mask.sum() == 1:
                stacked_df = stacked_df[~mask]
                session['stats_data'] = stacked_df.to_json(orient='records', double_precision=4)
            else:
                session['error'] = "Could not uniquely identify row to delete"
            return redirect(url_for('stats'))  # 🔁

        # VIEW stats (add row)
        elif 'view_stats' in request.form:
            player = request.form.get('player', '').upper().strip()
            year = request.form.get('year')
            event = request.form.get('event')
            table_name = event_table_map.get(year, {}).get(event)
            if not player or not table_name:
                session['error'] = "Please select a valid player and event."
            else:
                try:
                    conn = sqlite3.connect('/home/prlfive/DataCT/vct.db')
                    df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE player = ?", conn, params=(player,))
                    conn.close()
                    if df.empty:
                        session['error'] = f"No data found for player '{player}' in event '{event} {year}'."
                    else:
                        df.columns = [col.upper() for col in df.columns]
                        numeric_cols = ['RATING', 'ACS', 'K:D','CL','KPR', 'KAST', 'ADR', 'FKPR', 'FDPR', 'APR', 'HS','RAR']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = df[col].astype(float).round(4)
                        df['Event'] = f"{event} {year}"
                        if 'COUNTRY' in df.columns:
                            df = df.drop(columns=['COUNTRY'])
                        stacked_df = pd.concat([stacked_df, df], ignore_index=True)
                        for col in numeric_cols:
                            if col in stacked_df.columns:
                                stacked_df[col] = stacked_df[col].astype(float).round(2).apply(lambda x: f"{x:.2f}")
                        session['stats_data'] = stacked_df.to_json(orient='records', double_precision=4)
                except Exception as e:
                    session['error'] = f"Database error: {str(e)}"
            return redirect(url_for('stats'))  # 🔁

        # GENERATE graph
        elif 'generate_graph' in request.form:
            if stacked_df.empty:
                session['error'] = "No data available to generate graph."
            else:
                try:
                    metrics = ['RATING', 'ACS', 'K:D', 'KAST', 'ADR', 'FK/FD PR']
                    max_values = {'RATING': 2.0, 'ACS': 300, 'K:D': 2.0, 'KAST': 1.0, 'ADR': 200, 'FK/FD PR': 2.5}
                    players = stacked_df['PLAYER'] + ' (' + stacked_df['Event'] + ')'
                    data = stacked_df[ ['RATING', 'ACS', 'K:D', 'KAST', 'ADR', 'FKPR', 'FDPR'] ].astype(float).round(4)
                    data['FK/FD PR'] = (data['FKPR'] / data['FDPR']).replace([np.inf, -np.inf], max_values['FK/FD PR']).fillna(0).clip(upper=2.5)
                    data = data[metrics]
                    normalized_data = data.copy()
                    for metric in metrics:
                        # All metrics are now "higher is better", so normalization is direct
                        normalized_data[metric] = data[metric] / max_values[metric]

                    # --- Plotting Code (largely unchanged) ---
                    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() + [0]
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, polar=True)
                    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'black', 'cyan']
                    # Plot each player's data
                    for idx, label in enumerate(players):
                        values = normalized_data.iloc[idx].tolist() + [normalized_data.iloc[idx].tolist()[0]]
                        ax.plot(angles, values, color=colors[idx % len(colors)], linewidth=2, label=label)
                        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.25)

                    # Format the chart
                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(metrics)
                    plt.title('Player Performance Comparison', pad=30)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
                    plt.tight_layout()
                    plt.subplots_adjust(right=0.8)

                    # Save the figure
                    image_path = '/home/prlfive/DataCT/static/stats_plot.png'
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    plt.savefig(image_path)
                    plt.close()
                    session['graph_generated'] = True # use this in GET to show image

                except Exception as e:
                    session['error'] = f"Graph generation error: {str(e)}"
            return redirect(url_for('stats'))  # 🔁
        elif 'clear_stack' in request.form:
            session.pop('stats_data', None)
            session.pop('graph_generated', None)
            if os.path.exists('/home/prlfive/DataCT/static/stats_plot.png'):
                os.remove('/home/prlfive/DataCT/static/stats_plot.png')
            return redirect(url_for('stats'))  # 🔁

    # On GET: check if error or image needs to be shown
    error = session.pop('error', None)
    image_path = '/home/prlfive/DataCT/static/stats_plot.png' if session.pop('graph_generated', False) else None

    return render_template('stats.html',
        table=stacked_df,
        image_path=image_path,
        timestamp=datetime.now().timestamp() if image_path else None,
        error=error
    )



@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify([])

    conn = sqlite3.connect('/home/prlfive/DataCT/vct.db')
    cursor = conn.cursor()
    cursor.execute("""
    SELECT DISTINCT player FROM (
        SELECT player FROM stage1_24
        UNION SELECT player FROM stage1_25
        UNION SELECT player FROM stage1_23
    ) WHERE player LIKE ? LIMIT 10
    """, (query + '%',))
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify(results)

@app.template_filter('roundfloat')
def round_float_filter(value, precision=2):
    try:
        return round(float(value), precision)
    except (ValueError, TypeError):
        return value
if __name__ == '__main__':
    app.run(debug=False)
