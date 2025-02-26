import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_distance(point1, point2):
    """Calculate distance between two points."""
    return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_pitching_mechanics(frame, landmarks):
    """Analyze key aspects of pitching mechanics."""
    feedback = {}  # Use dictionary to store latest analysis for each metric
    
    if landmarks.landmark:
        # Shoulder alignment analysis
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_slope = (right_shoulder.y - left_shoulder.y) / (right_shoulder.x - left_shoulder.x)
        feedback['shoulder'] = "Shoulder alignment: Keep shoulders level through delivery" if abs(shoulder_slope) > 0.1 else "Shoulder alignment: Good level shoulders through delivery"

        # Hip rotation analysis
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_slope = (right_hip.y - left_hip.y) / (right_hip.x - left_hip.x)
        feedback['hip'] = "Hip rotation: Maintain proper hip alignment during delivery" if abs(hip_slope) > 0.1 else "Hip rotation: Good hip alignment through delivery"

        # Knee bend analysis
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        feedback['knee'] = "Knee bend: Good knee flexion" if knee_angle < 130 else "Knee bend: Increase knee bend for better power generation"

        # Stride length analysis
        stride_length = calculate_distance(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
                                        landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE])
        height_reference = calculate_distance(right_hip, right_ankle)
        stride_ratio = stride_length / height_reference
        
        if stride_ratio < 0.85:
            feedback['stride'] = "Stride Length: Too short - aim for 85-100% of height for optimal power transfer"
        elif stride_ratio > 1.1:
            feedback['stride'] = "Stride Length: Too long - may affect balance and control"
        else:
            feedback['stride'] = "Stride Length: Optimal range for power and control"

        # Trunk tilt analysis
        tilt_angle = math.degrees(math.atan2(
            landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x - landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
            landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y - landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        ))
        
        if abs(tilt_angle) < 20:
            feedback['trunk'] = "Trunk Tilt: Insufficient - aim for 20-30 degrees forward tilt"
        elif abs(tilt_angle) > 40:
            feedback['trunk'] = "Trunk Tilt: Excessive - reduce tilt to maintain balance"
        else:
            feedback['trunk'] = f"Trunk Tilt: Good forward tilt of {abs(tilt_angle):.1f} degrees"

        # Arm slot analysis
        arm_angle = calculate_angle(
            landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
            landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        )
        
        if arm_angle < 45:
            feedback['arm_slot'] = "Arm Slot: Low - may increase injury risk and reduce velocity"
        elif arm_angle < 60:
            feedback['arm_slot'] = "Arm Slot: Three-quarter - good for movement and deception"
        elif arm_angle < 85:
            feedback['arm_slot'] = "Arm Slot: High three-quarter - optimal for most pitchers"
        else:
            feedback['arm_slot'] = "Arm Slot: Over-the-top - good for downward plane, watch shoulder stress"

    return list(feedback.values())

def get_overall_summary(analysis_results):
    """Generate an overall summary of the pitching mechanics."""
    summary = {
        'strengths': [],
        'improvements': [],
        'overall_rating': 0,
        'priority_improvements': [],
        'drill_recommendations': {}
    }
    
    # Create unique set of feedback
    unique_feedback = list(set(analysis_results))
    
    # Define priority order for improvements
    priority_order = {
        'hip rotation': 1,      # Foundation of power and stability
        'knee bend': 2,         # Power generation
        'trunk tilt': 3,        # Core power transfer
        'shoulder alignment': 4, # Upper body mechanics
        'arm slot': 5,          # Arm action
        'stride length': 6      # Timing and distance
    }
    
    # Categorize feedback
    improvements = []
    for feedback in unique_feedback:
        is_positive = any(positive in feedback.lower() 
                        for positive in ['good', 'optimal', 'maintaining'])
        
        if is_positive:
            summary['strengths'].append(feedback)
        else:
            improvements.append(feedback)
    
    # Sort improvements by priority
    improvements.sort(key=lambda x: next((priority_order[k] for k in priority_order.keys() 
                                        if k in x.lower()), 999))
    summary['improvements'] = improvements
    
    # Calculate overall rating
    total_aspects = len(unique_feedback)
    if total_aspects > 0:
        summary['overall_rating'] = (len(summary['strengths']) / total_aspects) * 100
    
    # Add overall recommendation based on the most critical improvements
    if improvements:
        summary['priority_improvements'] = [
            "Focus on these key improvements in order:"
        ]
        
        # Add top 3 improvements with their drill recommendations
        for i, imp in enumerate(improvements[:3]):
            summary['priority_improvements'].append(f"{i+1}. {imp}")
            drills = get_drill_recommendations(imp)
            if drills:
                summary['drill_recommendations'][imp] = drills
    
    return summary

def get_drill_recommendations(improvement):
    """Get specific drill recommendations for each improvement area."""
    drills = {
        'hip rotation': [
            "• Hip/Core Separation Drill: Stand sideways with a resistance band, rotate hips while keeping shoulders stable",
            "• Medicine Ball Rotational Throws: Focus on hip drive and proper sequencing",
            "• Towel Drill: Practice hip rotation timing with a towel instead of throwing"
        ],
        'knee bend': [
            "• Drop & Drive Drill: Focus on back leg drive from athletic stance",
            "• Wall Sits: Strengthen legs in pitching position",
            "• Stride Drill with Pause: Hold knee lift position to reinforce proper bend"
        ],
        'trunk tilt': [
            "• Balance Drill: Practice proper tilt while maintaining balance on back leg",
            "• Mirror Work: Use mirror to check trunk angle during delivery",
            "• Resistance Band Forward Tilt: Practice controlled forward motion"
        ],
        'shoulder alignment': [
            "• Wall Slides: Keep shoulders level while sliding up/down wall",
            "• Band Pull-Aparts: Strengthen shoulder positioning",
            "• Balance Point Drill: Check shoulder alignment at key positions"
        ],
        'arm slot': [
            "• Wall Target Drill: Draw line on wall to practice consistent arm slot",
            "• Resistance Band High Pull: Reinforce proper arm path",
            "• Long Toss: Focus on maintaining slot during extended throwing"
        ],
        'stride length': [
            "• Measure & Mark Drill: Mark optimal stride length on ground",
            "• Stride Length Progression: Gradually increase stride length",
            "• Jump Drill: Practice explosive movement to proper distance"
        ]
    }
    
    for key, value in drills.items():
        if key in improvement.lower():
            return value
    return []

def find_release_point_frame(all_landmarks):
    """Find the frame index that best represents the release point."""
    max_extension = 0
    max_idx = 0
    
    for i, landmarks in enumerate(all_landmarks):
        # Calculate distance between shoulder and wrist
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        extension = calculate_distance(shoulder, wrist)
        
        if extension > max_extension:
            max_extension = extension
            max_idx = i
    
    return max_idx

def get_key_frames(frames, num_frames=8):
    """Select key frames from the pitching sequence."""
    if len(frames) < num_frames:
        return frames
    
    # Select frames at key points in the delivery
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return [frames[int(i)] for i in indices]  # Convert numpy index to int

def get_phase_description(frame_number):
    """Get description of pitching phase based on frame number."""
    phases = {
        1: "Set Position",
        2: "First Movement",
        3: "Maximum Knee Lift",
        4: "Stride",
        5: "Arm Cocking",
        6: "Acceleration",
        7: "Release Point",
        8: "Follow Through"
    }
    return phases.get(frame_number, "")

def process_video(video_file, progress_bar):
    """Process video file and analyze pitching mechanics."""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    # Initialize video capture
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    all_landmarks = []
    
    progress_bar.progress(0.1, "Loading video...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update progress for frame processing (40% of total progress)
        frame_count += 1
        progress = 0.1 + (frame_count / total_frames * 0.4)
        progress_bar.progress(progress, "Processing frames...")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            frames.append(frame_rgb)
            all_landmarks.append(results.pose_landmarks)
    
    cap.release()
    
    # Update progress for release point detection
    progress_bar.progress(0.6, "Detecting release point...")
    
    # Find the key frame for analysis
    if len(frames) > 0:
        # Find frame with maximum arm extension
        max_extension_idx = find_release_point_frame(all_landmarks)
        
        # Update progress for mechanics analysis
        progress_bar.progress(0.8, "Analyzing mechanics...")
        
        # Analyze mechanics at release point
        feedback = analyze_pitching_mechanics(frames[max_extension_idx], all_landmarks[max_extension_idx])
        
        # Complete the progress
        progress_bar.progress(1.0, "Analysis complete!")
        return frames, feedback
    
    progress_bar.progress(1.0, "Analysis complete!")
    return frames, []

# Streamlit UI
st.set_page_config(page_title="Baseball Pitcher Mechanics Analyzer", layout="wide")

st.title("Baseball Pitcher Mechanics Analyzer")
st.write("Upload a video of a pitcher from the side angle to analyze their mechanics.")

uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Analyze Mechanics"):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner(""):  # Empty spinner since we're using custom progress
            frames, analysis = process_video(uploaded_file, progress_bar)
            summary = get_overall_summary(analysis)
            
            # Clear the progress bar and status text after completion
            progress_bar.empty()
            status_text.empty()
            
            # Display Frame Analysis first
            st.header("Frame-by-Frame Analysis")
            if len(frames) > 0:
                # Display key frames in a grid
                key_frames = get_key_frames(frames, num_frames=8)
                
                # Create two rows of four frames each
                rows = [key_frames[i:i + 4] for i in range(0, len(key_frames), 4)]
                
                for row_idx, row in enumerate(rows):
                    cols = st.columns(4)
                    for col_idx, (col, frame) in enumerate(zip(cols, row)):
                        frame_number = row_idx * 4 + col_idx + 1
                        phase = get_phase_description(frame_number)
                        col.image(frame, caption=f"Phase {frame_number}: {phase}", use_column_width=True)
            
            # Then display the analysis summary
            st.header("Pitching Analysis Summary")
            
            # Display overall rating with context
            rating = summary['overall_rating']
            rating_color = "green" if rating >= 80 else "orange" if rating >= 60 else "red"
            st.markdown(f"### Overall Mechanics Rating: <span style='color:{rating_color}'>{rating:.1f}%</span>", unsafe_allow_html=True)
            
            # Add rating context
            if rating >= 80:
                st.success("Excellent mechanics! Focus on maintaining consistency.")
            elif rating >= 60:
                st.warning("Good foundation. Some adjustments needed for improvement.")
            else:
                st.error("Significant improvements needed. Focus on fundamental mechanics.")
            
            # Display strengths and improvements
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strengths")
                for strength in summary['strengths']:
                    st.markdown(f"✅ {strength}")
            
            with col2:
                st.subheader("Areas for Improvement")
                for improvement in summary['improvements']:
                    st.markdown(f"🔸 {improvement}")
            
            # Display priority improvements with drills
            if summary.get('priority_improvements'):
                st.subheader("Improvement Priority")
                for priority in summary['priority_improvements']:
                    st.markdown(f"**{priority}**")
                
                st.subheader("Recommended Drills")
                for imp, drills in summary['drill_recommendations'].items():
                    st.markdown(f"**For {imp}:**")
                    for drill in drills:
                        st.markdown(drill)
                    st.markdown("---")

            st.header("Detailed Analysis")
            for feedback in analysis:
                st.write(f"• {feedback}")

st.sidebar.header("About")
st.sidebar.write("""
This tool analyzes baseball pitching mechanics using computer vision and pose estimation.
It provides feedback on:
- Shoulder alignment
- Hip rotation
- Knee bend
- Stride length
- Trunk tilt
- Arm slot
- Overall body positioning
""")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a video of a pitcher from the side angle
2. Click 'Analyze Mechanics'
3. Review the feedback and analyzed frames
4. Make adjustments based on the suggestions
""")
