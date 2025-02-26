# Baseball Pitcher Mechanics Analyzer

This application analyzes baseball pitching mechanics from side-angle videos using computer vision and machine learning. It provides comprehensive feedback and analysis to help improve pitching performance.

## Features

### Mechanics Analysis
- **Overall Mechanics Rating**: 
  - Numerical score with color-coded feedback
  - Context-based recommendations
  - Performance level indicators

### Key Metrics Analysis
- **Fundamental Mechanics**:
  - Hip rotation and alignment
  - Knee bend and power generation
  - Trunk tilt and balance
  - Shoulder alignment
  - Arm slot positioning
  - Stride length optimization

### Prioritized Feedback
- **Strengths and Improvements**:
  - Clear separation of positive aspects and areas needing work
  - Prioritized list of improvements based on biomechanical importance
  - Visual indicators for easy reading ( strengths, improvements)

### Drill Recommendations
- **Customized Training Plan**:
  - Specific drills for each identified improvement area
  - Detailed instructions for each drill
  - Progressive development options
- **Common Drills Include**:
  - Hip/Core separation exercises
  - Power generation drills
  - Balance and stability work
  - Arm slot consistency training
  - Stride length optimization

### Visual Analysis
- **Frame-by-Frame Breakdown**: 
  - 6 key frames showing critical phases of delivery
  - Labeled phases from wind-up to follow-through
  - Real-time pose estimation overlay
  - Release point detection and analysis

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL
3. Upload a video of a pitcher from the side angle
4. Click "Analyze Mechanics" to receive:
   - Overall mechanics rating with context
   - Strengths and areas for improvement
   - Prioritized improvement list
   - Specific drill recommendations
   - Frame-by-frame analysis

## Deployment

### Docker Deployment (Recommended)

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```
   The application will be available at http://localhost:8501

2. For production deployment:
   ```bash
   docker build -t pitcher-mechanics-analyzer .
   docker run -p 8501:8501 pitcher-mechanics-analyzer
   ```

### Cloudflare Pages Deployment with Docker

1. Push this repository to GitHub
2. Log in to Cloudflare Dashboard
3. Go to Pages > Create a project
4. Connect your GitHub repository
5. Configure the build settings:
   - Framework preset: Docker
   - Build command: docker build -t pitcher-mechanics-analyzer . && docker run -d -p 8501:8501 pitcher-mechanics-analyzer
   - Build output directory: /
   - Environment variables:
     - DOCKER_BUILDKIT: 1

### Local Development (Without Docker)

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Requirements

- Python 3.8+
- Webcam (for live analysis)
- Video files in MP4, MOV, or AVI format

## Technical Details

The application uses:
- MediaPipe for advanced pose estimation
- OpenCV for video processing
- Streamlit for the modern web interface
- NumPy for mathematical calculations and angle analysis

## Best Practices for Video Recording

For optimal analysis:
1. Record from a direct side angle
2. Ensure good lighting conditions
3. Maintain a clear view of the entire pitching motion
4. Record at 60fps or higher if possible
5. Keep the camera stable during recording

## Analysis Methodology

The analyzer focuses on key phases of the pitching motion:
1. **Wind-up**: Initial balance and positioning
2. **First Movement**: Weight shift and timing
3. **Stride**: Length and direction
4. **Arm Cocking**: Shoulder positioning and arm slot
5. **Acceleration**: Power generation and trunk rotation
6. **Follow Through**: Balance and deceleration

Improvements are prioritized based on the kinetic chain:
1. Hip rotation (foundation)
2. Knee bend (power generation)
3. Trunk tilt (core power transfer)
4. Shoulder alignment
5. Arm slot
6. Stride length
