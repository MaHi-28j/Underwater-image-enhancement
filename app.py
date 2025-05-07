from PIL import Image, ImageStat, ImageFilter, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

def detect_color_cast(image):
    """
    Detect if the image has a blue or green color cast by analyzing the average values
    and distribution of color channels.
    Returns: (is_bluish, confidence_score, channel_ratios)
    """
    # Split the image into RGB channels
    r, g, b = image.split()
    
    # Calculate mean values for each channel
    r_mean = ImageStat.Stat(r).mean[0]
    g_mean = ImageStat.Stat(g).mean[0]
    b_mean = ImageStat.Stat(b).mean[0]
    
    # Calculate relative color intensities
    total = r_mean + g_mean + b_mean
    r_ratio = r_mean / total
    g_ratio = g_mean / total
    b_ratio = b_mean / total
    
    # Calculate confidence score based on difference between dominant and secondary color
    if b_ratio > g_ratio:
        confidence = (b_ratio - g_ratio) / b_ratio * 100
        is_bluish = True
    else:
        confidence = (g_ratio - b_ratio) / g_ratio * 100
        is_bluish = False
        
    channel_ratios = {
        'red': round(r_ratio * 100, 2),
        'green': round(g_ratio * 100, 2),
        'blue': round(b_ratio * 100, 2)
    }
    
    return is_bluish, confidence, channel_ratios

def compensate_RB(image, flag):
    """
    Compensate Red and Blue channels based on Green channel
    flag = 0: compensate both Red and Blue
    flag = 1: compensate only Red
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Split into channels
    imageR = img_array[:, :, 0].astype(np.float64)
    imageG = img_array[:, :, 1].astype(np.float64)
    imageB = img_array[:, :, 2].astype(np.float64)
    
    # Get dimensions
    y, x = imageR.shape
    
    # Get min and max values for each channel
    minR, maxR = imageR.min(), imageR.max()
    minG, maxG = imageG.min(), imageG.max()
    minB, maxB = imageB.min(), imageB.max()
    
    # Normalize to range [0, 1]
    imageR = (imageR - minR) / (maxR - minR + 1e-6)
    imageG = (imageG - minG) / (maxG - minG + 1e-6)
    imageB = (imageB - minB) / (maxB - minB + 1e-6)
    
    # Calculate means
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    
    # Compensate channels based on flag
    if flag == 0:  # Greenish image
        imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
        imageB = (imageB + (meanG - meanB) * (1 - imageB) * imageG) * maxB
        imageG = imageG * maxG
    else:  # Bluish image
        imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
        imageB = imageB * maxB
        imageG = imageG * maxG
    
    # Clip values to valid range
    imageR = np.clip(imageR, 0, 255).astype(np.uint8)
    imageG = np.clip(imageG, 0, 255).astype(np.uint8)
    imageB = np.clip(imageB, 0, 255).astype(np.uint8)
    
    # Combine channels
    compensateIm = np.dstack((imageR, imageG, imageB))
    
    return Image.fromarray(compensateIm)

def gray_world(image):
    """
    Apply gray world algorithm for white balancing
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Split channels
    imageR = img_array[:, :, 0].astype(np.float64)
    imageG = img_array[:, :, 1].astype(np.float64)
    imageB = img_array[:, :, 2].astype(np.float64)
    
    # Calculate means
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    meanGray = (meanR + meanG + meanB) / 3
    
    # Apply gray world algorithm
    imageR = np.clip(imageR * meanGray / meanR, 0, 255).astype(np.uint8)
    imageG = np.clip(imageG * meanGray / meanG, 0, 255).astype(np.uint8)
    imageB = np.clip(imageB * meanGray / meanB, 0, 255).astype(np.uint8)
    
    # Combine channels
    whitebalancedIm = np.dstack((imageR, imageG, imageB))
    
    return Image.fromarray(whitebalancedIm)

def sharpen(wbimage, original):
    """
    Sharpen the image using unsharp masking
    """
    # Create blurred version
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Convert to numpy arrays
    wb_array = np.array(wbimage)
    smooth_array = np.array(smoothed_image)
    
    # Apply unsharp mask formula
    sharpened = wb_array.astype(np.float64) * 2 - smooth_array.astype(np.float64)
    
    # Clip values to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return Image.fromarray(sharpened)

def hsv_global_equalization(image):
    """
    Perform global histogram equalization in HSV color space
    """
    # Convert to HSV
    hsv_image = image.convert('HSV')
    
    # Split channels
    h, s, v = hsv_image.split()
    
    # Equalize value channel
    v_eq = ImageOps.equalize(v)
    
    # Merge channels
    hsv_eq = Image.merge('HSV', (h, s, v_eq))
    
    # Convert back to RGB
    return hsv_eq.convert('RGB')

def average_fusion(image1, image2):
    """
    Fuse two images using simple averaging
    """
    # Convert to numpy arrays
    img1_array = np.array(image1).astype(np.float64)
    img2_array = np.array(image2).astype(np.float64)
    
    # Average the arrays
    fused_array = ((img1_array + img2_array) / 2).astype(np.uint8)
    
    return Image.fromarray(fused_array)

def pca_fusion(image1, image2):
    """
    Fuse two images using PCA-based fusion
    """
    # Convert images to numpy arrays
    img1_array = np.array(image1).astype(np.float64)
    img2_array = np.array(image2).astype(np.float64)
    
    # Process each channel separately
    result = np.zeros_like(img1_array)
    
    for i in range(3):  # For each color channel
        # Reshape the channel data
        x1 = img1_array[:, :, i].flatten()
        x2 = img2_array[:, :, i].flatten()
        
        # Stack the data
        X = np.vstack((x1, x2))
        
        # Calculate covariance matrix
        cov = np.cov(X)
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Calculate weights
        weights = eigenvecs[:, -1] / np.sum(eigenvecs[:, -1])
        
        # Apply weights and reshape
        fused_channel = (weights[0] * x1 + weights[1] * x2).reshape(img1_array.shape[:2])
        
        # Store result
        result[:, :, i] = fused_channel
    
    # Clip values to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)

def combined_fusion(image1, image2, weight_pca=0.6):
    """
    Combine PCA and average fusion results with a weighted approach
    """
    # Get PCA fusion result
    pca_result = pca_fusion(image1, image2)
    
    # Get average fusion result
    avg_result = average_fusion(image1, image2)
    
    # Convert to numpy arrays
    pca_array = np.array(pca_result).astype(np.float64)
    avg_array = np.array(avg_result).astype(np.float64)
    
    # Combine using weighted average
    combined_array = (weight_pca * pca_array + (1 - weight_pca) * avg_array)
    
    # Clip values to valid range
    combined_array = np.clip(combined_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(combined_array)

def display_color_analysis(image):
    """
    Display detailed color analysis of the image in the Streamlit interface
    """
    is_bluish, confidence, ratios = detect_color_cast(image)
    
    # Display color cast type
    if is_bluish:
        st.markdown("### 🔵 Detected: Bluish underwater image")
    else:
        st.markdown("### 🟢 Detected: Greenish underwater image")
    
    # Display confidence score
    st.write(f"Confidence: {confidence:.1f}%")
    
    # Create color bars for channel distribution
    st.write("Channel Distribution:")
    
    # Red channel
    st.markdown(
        f"""
        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
            <div style='width: 60px; margin-right: 10px;'>Red:</div>
            <div style='background-color: rgba(255,0,0,0.2); width: {ratios['red']}%; height: 20px; 
                 border-radius: 3px; margin-right: 10px;'></div>
            <div>{ratios['red']}%</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Green channel
    st.markdown(
        f"""
        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
            <div style='width: 60px; margin-right: 10px;'>Green:</div>
            <div style='background-color: rgba(0,255,0,0.2); width: {ratios['green']}%; height: 20px; 
                 border-radius: 3px; margin-right: 10px;'></div>
            <div>{ratios['green']}%</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Blue channel
    st.markdown(
        f"""
        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
            <div style='width: 60px; margin-right: 10px;'>Blue:</div>
            <div style='background-color: rgba(0,0,255,0.2); width: {ratios['blue']}%; height: 20px; 
                 border-radius: 3px; margin-right: 10px;'></div>
            <div>{ratios['blue']}%</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Add histograms for each channel
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
    
    # Split channels
    r, g, b = image.split()
    
    # Plot histograms
    ax1.hist(np.array(r).ravel(), bins=256, color='red', alpha=0.6)
    ax1.set_title('Red Channel')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(np.array(g).ravel(), bins=256, color='green', alpha=0.6)
    ax2.set_title('Green Channel')
    ax2.set_xlabel('Pixel Value')
    
    ax3.hist(np.array(b).ravel(), bins=256, color='blue', alpha=0.6)
    ax3.set_title('Blue Channel')
    ax3.set_xlabel('Pixel Value')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Channel Statistics")
        stats_r = ImageStat.Stat(r)
        stats_g = ImageStat.Stat(g)
        stats_b = ImageStat.Stat(b)
        
        st.write("Mean Values:")
        st.write(f"- Red: {stats_r.mean[0]:.1f}")
        st.write(f"- Green: {stats_g.mean[0]:.1f}")
        st.write(f"- Blue: {stats_b.mean[0]:.1f}")
    
    with col2:
        st.markdown("### Color Balance")
        rg_ratio = stats_r.mean[0] / stats_g.mean[0]
        rb_ratio = stats_r.mean[0] / stats_b.mean[0]
        gb_ratio = stats_g.mean[0] / stats_b.mean[0]
        
        st.write("Channel Ratios:")
        st.write(f"- Red/Green: {rg_ratio:.2f}")
        st.write(f"- Red/Blue: {rb_ratio:.2f}")
        st.write(f"- Green/Blue: {gb_ratio:.2f}")
    
    return is_bluish

def underwater_image_enhancement(image, flag, pca_weight=0.6):
    """
    Complete pipeline for underwater image enhancement with combined fusion
    """
    progress_text = "Operation Progress"
    progress_bar = st.progress(0)
    
    # Compensate image based on flag
    st.text("1. Compensating Red/Blue Channel Based on Green Channel...")
    compensatedimage = compensate_RB(image, flag)
    progress_bar.progress(20)
    
    # Apply gray world algorithm
    st.text("2. White Balancing using Grayworld Algorithm...")
    whitebalanced = gray_world(compensatedimage)
    progress_bar.progress(40)
    
    # Contrast enhancement
    st.text("3. Enhancing Contrast using Global Histogram Equalization...")
    contrastenhanced = hsv_global_equalization(whitebalanced)
    progress_bar.progress(60)
    
    # Sharpen
    st.text("4. Sharpening Image using Unsharp Masking...")
    sharpenedimage = sharpen(whitebalanced, image)
    progress_bar.progress(70)
    
    # Fusion operations
    st.text("5. Performing Image Fusion...")
    pcafused = pca_fusion(sharpenedimage, contrastenhanced)
    averagefused = average_fusion(sharpenedimage, contrastenhanced)
    combinedfused = combined_fusion(sharpenedimage, contrastenhanced, pca_weight)
    progress_bar.progress(100)
    
    return pcafused, averagefused, combinedfused

def main():
    st.set_page_config(
        page_title="Underwater Image Enhancement",
        page_icon="🌊",
        layout="wide"
    )
    
    st.title("Underwater Image Enhancement")
    st.write("""
    This application enhances underwater images by correcting color cast and improving visibility.
    Upload your image to get started!
    """)
    
    with st.sidebar:
        st.header("Settings")
        show_steps = st.checkbox("Show enhancement steps", value=True)
        show_analysis = st.checkbox("Show detailed analysis", value=True)
        pca_weight = st.slider(
            "PCA Weight in Combined Fusion",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            help="Weight of PCA fusion in the combined result (0: only average, 1: only PCA)"
        )
    
    uploaded_file = st.file_uploader(
        "Upload an underwater image (JPG or PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        try:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
            # Analyze and display color cast
            if show_analysis:
                with st.expander("📊 Color Analysis", expanded=True):
                    is_bluish = display_color_analysis(image)
            else:
                is_bluish = detect_color_cast(image)[0]
            
            # Enhancement process
            if st.button("🎨 Enhance Image"):
                with st.spinner("Enhancing image..."):
                    # Process based on detected color cast
                    flag = 1 if is_bluish else 0
                    
                    # Process image
                    pcafused, averagefused, combinedfused = underwater_image_enhancement(
                        image, flag, pca_weight
                    )
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs([
                        "Combined Fusion",
                        "PCA Fusion",
                        "Average Fusion"
                    ])
                    
                    with tab1:
                        st.subheader("Combined Fusion Result")
                        st.write(f"PCA Weight: {pca_weight:.2f}, Average Weight: {1-pca_weight:.2f}")
                        st.image(combinedfused, use_column_width=True)
                        if st.button("Download Combined Result", key="combined"):
                            combinedfused.save("enhanced_combined.png")
                            with open("enhanced_combined.png", "rb") as file:
                                st.download_button(
                                    "Download Combined Enhanced Image",
                                    data=file,
                                    file_name="enhanced_combined.png",
                                    mime="image/png"
                                )
                    
                    with tab2:
                        st.subheader("PCA Fusion Result")
                        st.image(pcafused, use_column_width=True)
                        if st.button("Download PCA Result", key="pca"):
                            pcafused.save("enhanced_pca.png")
                            with open("enhanced_pca.png", "rb") as file:
                                st.download_button(
                                    "Download PCA Enhanced Image",
                                    data=file,
                                    file_name="enhanced_pca.png",
                                    mime="image/png"
                                )
                    
                    with tab3:
                        st.subheader("Average Fusion Result")
                        st.image(averagefused, use_column_width=True)
                        if st.button("Download Average Result", key="avg"):
                            averagefused.save("enhanced_avg.png")
                            with open("enhanced_avg.png", "rb") as file:
                                st.download_button(
                                    "Download Average Enhanced Image",
                                    data=file,
                                    file_name="enhanced_avg.png",
                                    mime="image/png"
                                )
                    
                    # Add comparison view
                    st.subheader("Side-by-Side Comparison")
                    cols = st.columns(4)
                    with cols[0]:
                        st.write("Original")
                        st.image(image)
                    with cols[1]:
                        st.write("Combined Fusion")
                        st.image(combinedfused)
                    with cols[2]:
                        st.write("PCA Fusion")
                        st.image(pcafused)
                    with cols[3]:
                        st.write("Average Fusion")
                        st.image(averagefused)
                    
                st.success("Enhancement completed!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try uploading a different image.")
    
    else:
        st.info("Please upload an underwater image to begin")

if __name__ == "__main__":
    main()