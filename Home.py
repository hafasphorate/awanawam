import streamlit as st

# Must be the very first Streamlit command called on the page
st.set_page_config(
    page_title="Spatial & Crowd Dynamics Toolkit",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏗️ Spatial & Crowd Dynamics Toolkit")
st.markdown("""
## Welcome to Awanawam!

**Designing Intuitive Crowd Management Interventions using Urban Data and Simulations**

This toolkit provides integrated spatial analysis and computer vision tools for architects, urban planners, and researchers to understand how spatial configuration impacts crowd movement, behavior, and dynamics in buildings and urban spaces.
""")

# ============================================================================
# QUICK START GUIDE
# ============================================================================
st.markdown("---")
st.markdown("""
## 🚀 Quick Start Guide

### Step 1: Choose Your Analysis Type
Select from the modules in the sidebar to begin your analysis. Each module is designed for a specific type of spatial research.

### Step 2: Upload Your Data
- **Floorplans:** Use DXF or DWG CAD files from AutoCAD, Revit, or similar software
- **Videos:** MP4/AVI surveillance footage of crowd movements
- **Data Files:** JSON or CSV files with spatial/crowd metrics

### Step 3: Analyze & Visualize
Each module provides interactive visualizations, metrics calculations, and exportable results for further research.

---

## 📚 Module Guide

### 1️⃣ Visibility Graph Analysis (VGA)
**What it does:** Analyzes how visible different zones are from each point in a floorplan.

**Key Metrics:**
- **Visual Integration:** How connected a location is to other visible areas
- **Entropy:** Randomness/complexity of the visibility field
- **Isovist Polygon:** The visible area from a single point
- **Connectivity:** How many zones can "see" each other

**How to use:**
1. Upload a DXF or DWG floorplan file
2. Adjust grid size (spacing of analysis points) and ray angle step (precision)
3. Click inside room/corridor zones to highlight them (green selection)
4. Run analysis to calculate spatial metrics for selected areas
5. Export results as JSON or CSV
6. Optionally save your session to reload later

**Best for:** Interior design, retail store layouts, museum exhibits, office productivity analysis

---

### 2️⃣ Urban Space Syntax Analysis
**What it does:** Analyzes urban street networks to identify primary movement corridors and spatial integration.

**Key Metrics:**
- **Betweenness Centrality:** Percentage of all shortest paths that pass through a street segment (proxy for movement/foot traffic)
- **Network Choice:** Streets that are "shortcuts" between distant areas
- **Spatial Integration:** How easily locations connect to the rest of the network

**How to use:**
1. Enter a location (e.g., "Tiong Bahru, Singapore")
2. Set analysis radius (300-2000 meters)
3. Choose network type: walk, drive, or all
4. Click "Run Axial Analysis"
5. View interactive map showing street segments colored by betweenness centrality
6. Red nodes highlight high-contrast intersections

**Best for:** Urban planning, traffic studies, walkability analysis, retail location analysis

---

### 3️⃣ Video Homography & Tracking
**What it does:** Tracks human movement from surveillance videos and projects real-world coordinates onto your floorplan.

**Key Steps:**
1. **Import Data (Tab 3.1):**
   - Upload surveillance video (MP4/AVI)
   - Upload corresponding floorplan (DXF/DWG)
   - Or load a previously saved session

2. **Define ROI & Masking (Tab 3.2):**
   - Select 4 corner points on your floorplan (these establish the coordinate mapping)
   - These corners define how video pixels map to real-world coordinates
   - Optional: Draw exclusion masks to ignore areas (e.g., reflections, occlusions)

3. **Occupancy Analytics (Tab 3.3):**
   - Runs YOLO-based person detection on video frames
   - Tracks individuals across frames using YOLOv8 tracking
   - Applies homography transformation to project tracking points onto floorplan
   - Generates occupancy heatmaps showing where people spend time
   - Calculates metrics: crowd density, dwell time, movement speed

4. **2D Playback (Tab 3.4):**
   - Replay tracked movement overlaid on floorplan
   - View crowd density heatmaps
   - Use frame slider to navigate; click Play button in animated charts for continuous replay
   - Export results for further analysis

**Model Options:**
- **yolov8n.pt:** Fast, CPU-friendly (recommended for cloud)
- **yolov8s.pt:** Balanced accuracy and speed
- **rtdetr-l.pt:** High accuracy, transformer-based
- **yolov8x-pose.pt:** Includes pose/keypoint tracking

**Best for:** Crowd flow studies, event planning, evacuation analysis, retail foot traffic

---

### 4️⃣ Spatial vs. Crowd Data Correlation
**What it does:** Statistical analysis linking spatial metrics (from VGA) to crowd behavior metrics (from video tracking).

**Analysis Types:**
- **Pearson Correlation:** Strength of linear relationships between variables
- **Spearman Correlation:** Monotonic (non-linear) relationships
- **Pairwise Correlation Matrix:** Visual heatmap of all metric relationships

**How to use:**
1. Upload a combined JSON file containing:
   - VGA floorplan node data (visual integration, entropy, connectivity)
   - Crowd metrics (density, dwell time, speed)
2. Select which columns to analyze
3. Choose correlation method (Pearson or Spearman)
4. View correlation matrix with:
   - **Diagonal:** Distribution curves for each metric
   - **Lower triangle:** Scatter plots showing relationships
   - **Upper triangle:** Color-coded correlation coefficients

**Example Insights:**
- "High visual integration correlates with higher crowd density"
- "Dead-end corridors have lower foot traffic but higher dwell times"

**Best for:** Research papers, evidence-based design validation, hypothesis testing

---

### 5️⃣ Aggregated Insights
**What it does:** Synthesizes data from all case studies stored in the central database to identify patterns and trends across multiple buildings/spaces.

**Features:**
- Connects to Supabase database for collaborative research
- Aggregates all user-contributed VGA and crowd data
- Calculates correlation trends across multiple case studies
- Identifies common patterns in spatial configuration vs. crowd behavior

**How to use:**
1. Page automatically fetches latest aggregated data from database (10-min cache)
2. View correlation matrix of all spatial metrics vs. crowd metrics
3. Explore how different building types compare
4. Contribute your own case study results to expand collective dataset

**Best for:** Meta-analysis, trend identification, publishing research, validating spatial theory

---

## 🎯 Typical Workflow

### For Architects/Designers:
1. Start with Module 1 (VGA) to analyze your floorplan
2. Use Module 4 to compare metrics to existing crowd data
3. Iterate design based on spatial insights

### For Researchers:
1. Collect surveillance video data (Module 3)
2. Analyze floorplan with VGA metrics (Module 1)
3. Run correlation analysis (Module 4)
4. Publish findings; contribute to Module 5 aggregated database

### For Urban Planners:
1. Use Module 2 for street network analysis
2. Identify high-betweenness corridors for intervention
3. Plan crowd management or accessibility improvements

---

## 📊 Data Formats

### Floorplans
- **Format:** DXF or DWG (AutoCAD-compatible)
- **Content:** Line entities representing walls, furniture, boundaries
- **Export from:** AutoCAD, Revit, SketchUp, LibreCAD

### Videos
- **Format:** MP4, AVI, MOV
- **Content:** Surveillance footage showing crowd movement
- **Requirements:** Clear view of space, sufficient lighting

### Session Files (JSON)
```json
{
  "vga_floorplan_nodes": [
    {
      "x": 1000,
      "y": 2000,
      "isovist_area": 45000,
      "isovist_perimeter": 1200,
      "visual_integration": 0.85,
      "entropy": 4.2,
      "crowd_density": 0.3
    }
  ]
}
```

---

## ⚙️ Tips for Best Results

### VGA Analysis
- **Grid Size:** Smaller grids (200-500mm) = more detail but slower computation
- **Ray Step:** Smaller angles (1-2°) = higher precision but slower
- **Selection:** Click inside distinct zones; avoid corners/walls

### Video Tracking
- **Video Quality:** Higher resolution = better detection accuracy
- **Lighting:** Ensure consistent, adequate lighting throughout video
- **Corners:** Select 4 corners that define the boundaries of your analysis area clearly
- **Confidence Threshold:** Lower for crowded scenes, higher for sparse crowds

### Correlation Analysis
- **Sample Size:** More nodes = more robust correlations
- **Data Quality:** Ensure both spatial and crowd metrics are complete and accurate
- **Outliers:** Review extreme values; they may indicate data collection issues

---

## 🔗 Integration

### Supabase Database
Module 5 connects to a Supabase database for collaborative research:
- Requires `SUPABASE_URL` and `SUPABASE_KEY` in `secrets.toml`
- Stores anonymized case study data
- Enables cross-organization research collaboration

### Configuration
Create `.streamlit/secrets.toml`:
```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

---

## 📝 Tips & Interactions

- **VGA Selection:** Click inside a zone to select; hold and drag to pan
- **Video Playback:** Use frame slider to navigate; click Play button in animated charts for continuous replay (Plotly feature)
- **Exports:** Right-click Plotly charts to download as PNG or SVG (Plotly built-in feature)
- **Module Navigation:** Use the sidebar dropdown to select different modules

---

## ❓ FAQ

**Q: Can I use DWG files directly?**  
A: DWG conversion requires system tools (dwg2dxf or ODA). If unavailable, export your DWG as DXF in your CAD software.

**Q: How accurate is video tracking?**  
A: Accuracy depends on video quality, lighting, and model choice. YOLO typically achieves 85-95% detection rate in good conditions.

**Q: Can I combine multiple videos of the same space?**  
A: Yes! Export tracking results as CSV and manually merge before correlation analysis.

**Q: Is this tool suitable for real-time analysis?**  
A: No, this is designed for post-hoc analysis. Processing takes minutes to hours depending on data size.

---

## 📧 Support & Feedback

For issues, questions, or feature requests, please contact the development team.

**Happy analyzing! 🎉**
""")