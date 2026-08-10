# ==========================================
# TAB 4: 2D PLAYBACK & CROWD HEATMAPS
# ==========================================
with tab_playback:
    st.subheader("Step 2.4: 2D Playback & Crowd Trajectory Analytics")

    st.markdown("### 1. Import Tracking Dataset")
    col_up1, col_up2 = st.columns(2)

    def parse_tracking_json(raw_json):
        if isinstance(raw_json, list):
            return pd.DataFrame(raw_json)

        if isinstance(raw_json, dict):
            for key in ["tracking_points", "tracking_results", "pedestrian_trajectories", "trajectories", "tracking_data"]:
                if key in raw_json and isinstance(raw_json[key], list) and len(raw_json[key]) > 0:
                    return pd.DataFrame(raw_json[key])

        return pd.json_normalize(raw_json)

    with col_up1:
        uploaded_tb_json = st.file_uploader("Upload JSON Export (from Step 2.3)", type=["json"], key="tb_json_up")
        if uploaded_tb_json is not None:
            try:
                raw_json = json.load(uploaded_tb_json)
                df_loaded = parse_tracking_json(raw_json)
                st.session_state.tracking_results_df = df_loaded
                st.success(f"✅ Successfully imported {len(df_loaded)} tracking records!")
            except Exception as e:
                st.error(f"Error reading JSON: {e}")

    with col_up2:
        uploaded_tb_csv = st.file_uploader("Upload CSV Tracking Export", type=["csv"], key="tb_csv_up")
        if uploaded_tb_csv is not None:
            try:
                st.session_state.tracking_results_df = pd.read_csv(uploaded_tb_csv)
                st.success("✅ Successfully imported CSV tracking records!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.markdown("---")

    df_track = st.session_state.get("tracking_results_df")

    if df_track is not None and not df_track.empty:
        df_track.columns = [str(c).lower().strip() for c in df_track.columns]

        frame_col = next((c for c in ["frame_idx", "frame", "frame_number", "timestamp"] if c in df_track.columns), None)
        x_col = next((c for c in ["world_x", "x", "x (m)", "x_m", "pos_x", "x_canvas", "img_x"] if c in df_track.columns), None)
        y_col = next((c for c in ["world_y", "y", "y (m)", "y_m", "pos_y", "y_canvas", "img_y"] if c in df_track.columns), None)
        id_col = next((c for c in ["track_id", "id", "person_id"] if c in df_track.columns), "track_id")

        if x_col and y_col:
            if not frame_col:
                df_track["frame_idx"] = 0
                frame_col = "frame_idx"

            if id_col not in df_track.columns:
                df_track[id_col] = 1

            if "speed" not in df_track.columns:
                df_track = df_track.sort_values(by=[id_col, frame_col])
                df_track["dx"] = df_track.groupby(id_col)[x_col].diff().fillna(0)
                df_track["dy"] = df_track.groupby(id_col)[y_col].diff().fillna(0)
                df_track["speed"] = np.sqrt(df_track["dx"]**2 + df_track["dy"]**2)

            st.markdown("### 2. Motion Playback & Frame Analytics")
            frames_available = sorted(df_track[frame_col].unique())
            selected_f = st.slider(
                "Select Frame for Instant Inspection",
                min_value=int(min(frames_available)),
                max_value=int(max(frames_available)),
                value=int(min(frames_available)),
            )

            curr_frame_df = df_track[df_track[frame_col] == selected_f]

            col_fb1, col_fb2 = st.columns(2)

            with col_fb1:
                st.markdown(f"**Pedestrian Plan View (Frame #{selected_f})**")
                fig_play = go.Figure()
                fig_play = add_cad_walls_to_fig(fig_play)

                fig_play.add_trace(go.Scatter(
                    x=curr_frame_df[x_col],
                    y=curr_frame_df[y_col],
                    mode="markers+text",
                    marker=dict(size=12, color="#FF5722"),
                    text=curr_frame_df[id_col].astype(str),
                    textposition="top center",
                    name="Pedestrians"
                ))
                fig_play.update_layout(
                    template="plotly_dark", height=400, margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(scaleanchor="y", scaleratio=1)
                )
                st.plotly_chart(fig_play, use_container_width=True)

            with col_fb2:
                st.markdown(f"**Instant Density Heatmap (Frame #{selected_f})**")
                fig_f_hm = go.Figure()
                fig_f_hm = add_cad_walls_to_fig(fig_f_hm, wall_color="#FFFFFF", width=2)

                fig_f_hm.add_trace(go.Histogram2dContour(
                    x=curr_frame_df[x_col],
                    y=curr_frame_df[y_col],
                    colorscale="Jet",
                    showscale=True
                ))
                fig_f_hm.update_layout(
                    template="plotly_dark", height=400, margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(scaleanchor="y", scaleratio=1)
                )
                st.plotly_chart(fig_f_hm, use_container_width=True)

            st.markdown("---")

            st.markdown("### 3. Aggregated Crowd Metrics (Entire Video)")

            m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs([
                "📊 Crowd Volume", "🔥 Density Heatmap", "⚡ Speed Distribution", "🧭 Directional Flow"
            ])

            with m_tab1:
                st.markdown("#### Cumulative Occupancy Heatmap")
                fig_vol = go.Figure()
                fig_vol = add_cad_walls_to_fig(fig_vol, wall_color="#FFFFFF", width=2)
                fig_vol.add_trace(go.Histogram2dContour(
                    x=df_track[x_col], y=df_track[y_col], colorscale="Viridis", showscale=True
                ))
                fig_vol.update_layout(template="plotly_dark", height=500, xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_vol, use_container_width=True)

            with m_tab2:
                st.markdown("#### Binned Pedestrian Density Grid")
                fig_dens = go.Figure()
                fig_dens = add_cad_walls_to_fig(fig_dens, wall_color="#FFFFFF", width=2)
                fig_dens.add_trace(go.Histogram2d(
                    x=df_track[x_col], y=df_track[y_col], colorscale="Hot", showscale=True, nbinsx=35, nbinsy=35
                ))
                fig_dens.update_layout(template="plotly_dark", height=500, xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_dens, use_container_width=True)

            with m_tab3:
                st.markdown("#### Velocity Heatmap")
                fig_spd = px.scatter(
                    df_track, x=x_col, y=y_col, color="speed", color_continuous_scale="Plasma",
                    title="Pedestrian Speed Distribution"
                )
                fig_spd = add_cad_walls_to_fig(fig_spd)
                fig_spd.update_layout(template="plotly_dark", height=500, xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_spd, use_container_width=True)

            with m_tab4:
                st.markdown("#### Movement Direction Vectors")
                fig_dir = px.scatter(
                    df_track, x=x_col, y=y_col, color="dx", color_continuous_scale="RdBu",
                    title="Directional Shift Field (dx)"
                )
                fig_dir = add_cad_walls_to_fig(fig_dir)
                fig_dir.update_layout(template="plotly_dark", height=500, xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_dir, use_container_width=True)

            st.markdown("---")
            st.markdown("### 4. Export Aggregated Analytics")

            crowd_metrics_export = {
                "total_frames": int(df_track[frame_col].nunique()),
                "total_unique_pedestrians": int(df_track[id_col].nunique()),
                "average_speed": float(df_track["speed"].mean()),
                "max_speed": float(df_track["speed"].max()),
                "trajectories": df_track[[frame_col, id_col, x_col, y_col, "speed"]].to_dict(orient="records")
            }

            st.download_button(
                label="💾 Export Analytics JSON",
                data=json.dumps(crowd_metrics_export, indent=2),
                file_name="crowd_analytics.json",
                mime="application/json",
                use_container_width=True,
            )

        else:
            st.error(f"⚠️ Could not resolve coordinate columns in dataset. Found columns: {list(df_track.columns)}")

    else:
        st.info("💡 Upload a JSON/CSV tracking file above or run tracking in Step 2.3 to view movement playback and heatmaps.")