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
app.secret_key = 'your-secret-key'  # Needed for session usage

event_table_map = {
    '2023': {'lockin': 'lockin_23', 'stage1': 'stage1_23', 'tokyo': 'tokyo_23', 'champs': 'champs_23'},
    '2024': {'kickoff': 'kickoff_24', 'stage1': 'stage1_24', 'madrid': 'madrid_24', 'stage2': 'stage2_24', 'champs': 'champs_24'},
    '2025': {'kickoff': 'kickoff_25', 'stage1': 'stage1_25', 'bangkok': 'bangkok_25'}
}
home_map = {
    'kickoff_25': '2025 Kickoff',
    'bangkok_25': '2025 Masters Bangkok',
    'stage1_25': '2025 Stage1',
}
@app.route('/')
def home():
    top5 = []
    random_player = None
    try:
        conn = sqlite3.connect('E:/dataCT/vct.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get top 5 players by rating from stage1_25
        cursor.execute("SELECT player, team, rating, acs, adr FROM stage1_25 ORDER BY rating DESC LIMIT 5")
        top5 = cursor.fetchall()

        # Get random player from any of kickoff_25, bangkok_25, stage1_25 where rating > 1.0
        candidates = []
        for table in ['kickoff_25', 'bangkok_25', 'stage1_25']:
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

    # Load data from session if exists
    if 'stats_data' in session:
        try:
            stacked_df = pd.read_json(session['stats_data'])
            # Ensure numeric columns are properly typed and rounded
            numeric_cols = ['RATING', 'ACS', 'K:D', 'KAST', 'ADR', 'FKPR', 'FDPR', 'APR', 'HS']
            for col in numeric_cols:
                if col in stacked_df.columns:
                    stacked_df[col] = stacked_df[col].astype(float).round(4)
        except:
            session.pop('stats_data', None)

    if request.method == 'POST':
        if 'delete_row' in request.form:
            try:
                delete_player = request.form.get('delete_player').strip()
                delete_event = request.form.get('delete_event').strip()

                # Create a mask with exact matches (case-sensitive)
                mask = (stacked_df['PLAYER'].str.strip() == delete_player) & \
                       (stacked_df['Event'].str.strip() == delete_event)

                # Only delete if we find exactly one match
                if mask.sum() == 1:
                    stacked_df = stacked_df[~mask]
                    session['stats_data'] = stacked_df.to_json(orient='records', double_precision=4)
                else:
                    error = "Could not uniquely identify row to delete"

            except Exception as e:
                error = f"Error deleting row: {str(e)}"

        elif 'view_stats' in request.form:
            player = request.form.get('player', '').upper().strip()
            year = request.form.get('year')
            event = request.form.get('event')
            table_name = event_table_map.get(year, {}).get(event)

            if not player or not table_name:
                error = "Please select a valid player and event."
            else:
                try:
                    conn = sqlite3.connect('vct.db')
                    query = f"SELECT * FROM {table_name} WHERE player = ?"
                    df = pd.read_sql_query(query, conn, params=(player,))
                    conn.close()

                    if df.empty:
                        error = f"No data found for player '{player}' in event '{event} {year}'."
                    else:
                        # Standardize column names and round numeric values
                        df.columns = [col.upper() for col in df.columns]
                        numeric_cols = ['RATING', 'ACS', 'K:D', 'KAST', 'ADR', 'FKPR', 'FDPR', 'APR', 'HS']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = df[col].astype(float).round(4)

                        df['Event'] = f"{event} {year}"
                        if 'COUNTRY' in df.columns:
                            df = df.drop(columns=['COUNTRY'])

                        stacked_df = pd.concat([stacked_df, df], ignore_index=True)
                        session['stats_data'] = stacked_df.to_json(orient='records', double_precision=4)
                except Exception as e:
                    error = f"Database error: {str(e)}"

        elif 'generate_graph' in request.form:
            if stacked_df.empty:
                error = "No data available to generate graph."
            else:
                try:
                    metrics = ['RATING', 'ACS', 'K:D', 'KAST', 'ADR']
                    if 'RAR' in stacked_df.columns and not stacked_df['RAR'].isnull().all():
                        metrics.append('RAR')

                    # Create normalized data with rounding
                    normalized_data = pd.DataFrame()
                    for metric in metrics:
                        series = stacked_df[metric].astype(float).round(4)
                        min_val = series.min()
                        max_val = series.max()

                        if min_val == max_val:
                            normalized_data[metric] = [0.5] * len(series)
                        else:
                            normalized_data[metric] = ((series - min_val) / (max_val - min_val)).round(4)

                    # Radar chart generation (same as before)
                    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                    angles += angles[:1]

                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, polar=True)

                    labels = stacked_df['PLAYER'] + ' (' + stacked_df['Event'] + ')'
                    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'black', 'cyan']

                    for idx, label in enumerate(labels):
                        values = normalized_data.iloc[idx].tolist()
                        values += values[:1]
                        ax.plot(angles, values, color=colors[idx % len(colors)], linewidth=2, label=label)
                        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.25)

                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(metrics)
                    ax.set_rlabel_position(0)
                    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], color="grey", size=7)
                    plt.ylim(0, 1)

                    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Players', frameon=True,
                              edgecolor='black')
                    plt.title('Event Stats Comparison', pad=30)
                    plt.tight_layout()
                    plt.subplots_adjust(right=0.8)

                    image_path = 'static/stats_plot.png'
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    plt.savefig(image_path)
                    plt.close()

                except Exception as e:
                    error = f"Graph generation error: {str(e)}"

        elif 'clear_stack' in request.form:
            session.pop('stats_data', None)
            if os.path.exists('static/stats_plot.png'):
                os.remove('static/stats_plot.png')
            stacked_df = pd.DataFrame()

    return render_template('stats.html',table=stacked_df,image_path=image_path,timestamp=datetime.now().timestamp() if image_path else None,
                               error=error)


@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify([])

    conn = sqlite3.connect('vct.db')
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
    app.run(debug=True)