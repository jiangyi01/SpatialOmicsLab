def _safe_filename_part(text):
    """``text`` made safe to embed in a single output filename.

    Subject and sample identifiers arrive as free text, and a '/' in one turned the output name into a
    path through a directory that does not exist -- so the save failed after the analysis had already
    run (hunt 2026-09-30, uT7-immunology-28).
    """
    import re

    return re.sub(r"[^\w.-]+", "_", str(text)).strip(".") or "unnamed"


def reconstruct_3d_face_from_mri(mri_file_path, output_dir="./output", subject_id="subject", threshold_value=300):
    """Generate a 3D model of facial anatomy from MRI scans of the head and neck.

    Parameters
    ----------
    mri_file_path : str
        Path to the MRI scan file (NIfTI format: .nii or .nii.gz)
    output_dir : str
        Directory where output files will be saved
    subject_id : str
        Identifier for the subject, used in output filenames
    threshold_value : int
        Threshold value for initial segmentation of facial tissues

    Returns
    -------
    str
        Research log detailing the reconstruction process and output file locations

    """
    import os
    import time
    from datetime import datetime

    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    from skimage import measure

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    log = []
    log.append(f"3D Facial Reconstruction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"Subject ID: {subject_id}")
    log.append(f"MRI Source: {mri_file_path}")
    log.append("-" * 50)

    # Step 1: Load MRI data
    start_time = time.time()
    log.append("Step 1: Loading MRI data")

    # Kept so the segmentation can be written back with the geometry it was read with (below).
    mri_data = mri_image = None
    try:
        # Try loading with nibabel first
        mri_data = nib.load(mri_file_path)
        volume_data = mri_data.get_fdata()
        log.append(f"  Loaded MRI data with dimensions: {volume_data.shape}")
        log.append(f"  Data type: {volume_data.dtype}")
    except Exception as e:
        log.append(f"  Error loading with nibabel: {str(e)}")
        log.append("  Attempting to load with SimpleITK...")
        mri_data = None

        try:
            # Fallback to SimpleITK
            mri_image = sitk.ReadImage(mri_file_path)
            volume_data = sitk.GetArrayFromImage(mri_image)
            log.append(f"  Loaded MRI data with dimensions: {volume_data.shape}")
        except Exception as e2:
            log.append(f"  Failed to load MRI data: {str(e2)}")
            return "\n".join(log)

    elapsed = time.time() - start_time
    log.append(f"  Completed in {elapsed:.2f} seconds")

    # Step 2: Preprocess MRI data
    start_time = time.time()
    log.append("Step 2: Preprocessing MRI data")

    # Normalize data to 0-1 range for processing
    data_min = np.min(volume_data)
    data_max = np.max(volume_data)
    # threshold_value is in the scan's own intensity units, so it has to fall inside the data range. The
    # default 300 on an 8-bit or [0, 1] scan gave an empty mask, and the run then died on an unassigned
    # mesh_file (hunt 2026-09-30, uT7-immunology-12).
    if not data_min < threshold_value < data_max:
        log.append(
            f"  Error: threshold_value={threshold_value} is outside this scan's intensity range "
            f"({data_min:g} to {data_max:g}); pass a threshold_value inside that range. No segmentation was made."
        )
        return "\n".join(log)
    normalized_data = (volume_data - data_min) / (data_max - data_min)

    # Apply noise reduction filter using SimpleITK
    sitk_image = sitk.GetImageFromArray(normalized_data)
    smoothed_image = sitk.CurvatureFlow(image1=sitk_image, timeStep=0.125, numberOfIterations=5)
    smoothed_data = sitk.GetArrayFromImage(smoothed_image)

    log.append("  Applied noise reduction filter")
    elapsed = time.time() - start_time
    log.append(f"  Completed in {elapsed:.2f} seconds")

    # Step 3: Segment facial tissues
    start_time = time.time()
    log.append("Step 3: Segmenting facial tissues")

    # Apply threshold to segment facial tissues (mapped the same way the data was normalised)
    normalized_threshold = (threshold_value - data_min) / (data_max - data_min)
    binary_mask = smoothed_data > normalized_threshold

    # Remove small isolated regions
    sitk_mask = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    cleaned_mask = sitk.BinaryOpeningByReconstruction(sitk_mask, [3, 3, 3])
    segmentation = sitk.GetArrayFromImage(cleaned_mask)

    log.append(f"  Applied threshold value: {threshold_value}")
    log.append("  Removed small isolated regions")

    # Save segmentation mask with the source geometry. GetImageFromArray alone dropped the affine and
    # read nibabel's (x, y, z) array as (z, y, x), so the mask did not overlay its scan (uT7-immunology-12).
    segmentation_file = os.path.join(output_dir, f"{_safe_filename_part(subject_id)}_face_segmentation.nii.gz")
    if mri_data is not None:
        nib.save(nib.Nifti1Image(segmentation.astype(np.uint8), mri_data.affine), segmentation_file)
    else:
        segmentation_image = sitk.GetImageFromArray(segmentation.astype(np.uint8))
        segmentation_image.CopyInformation(mri_image)
        sitk.WriteImage(segmentation_image, segmentation_file)
    log.append(f"  Saved segmentation mask to: {segmentation_file}")

    elapsed = time.time() - start_time
    log.append(f"  Completed in {elapsed:.2f} seconds")

    # Step 4: Generate 3D surface mesh
    start_time = time.time()
    log.append("Step 4: Generating 3D surface mesh")

    # Create mesh using marching cubes algorithm
    mesh_file = None
    try:
        verts, faces, normals, values = measure.marching_cubes(segmentation, level=0.5)
        log.append(f"  Generated mesh with {len(verts)} vertices and {len(faces)} faces")

        # Save mesh as OBJ file
        mesh_path = os.path.join(output_dir, f"{_safe_filename_part(subject_id)}_face_3d_model.obj")

        with open(mesh_path, "w") as f:
            # Write vertices
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write vertex normals
            for n in normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")

            # Write faces (OBJ uses 1-indexed vertices)
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        mesh_file = mesh_path
        log.append(f"  Saved 3D model to: {mesh_file}")
    except Exception as e:
        log.append(f"  Error generating mesh: {str(e)}")

    elapsed = time.time() - start_time
    log.append(f"  Completed in {elapsed:.2f} seconds")

    # Final summary
    log.append("-" * 50)
    log.append("Reconstruction Summary:")
    log.append(f"  Subject ID: {subject_id}")
    log.append(f"  Segmentation file: {os.path.basename(segmentation_file)}")
    log.append(
        f"  3D Model file: {os.path.basename(mesh_file)}"
        if mesh_file
        else "  3D Model file: none -- mesh generation failed (see Step 4)"
    )
    log.append("-" * 50)

    return "\n".join(log)


def analyze_abr_waveform_p1_metrics(time_ms, amplitude_uv):
    """Extracts P1 amplitude and latency from Auditory Brainstem Response (ABR) waveform data.

    P1 (Wave I) is typically the first positive peak in the ABR waveform and is a critical
    marker for auditory function assessment.

    Parameters
    ----------
    time_ms : array-like
        Time points of the ABR recording in milliseconds
    amplitude_uv : array-like
        Amplitude values of the ABR recording in microvolts

    Returns
    -------
    str
        Research log summarizing the analysis steps and results, including P1 amplitude and latency

    """
    import numpy as np
    from scipy.signal import find_peaks

    # Convert inputs to numpy arrays if they aren't already
    time_ms = np.array(time_ms)
    amplitude_uv = np.array(amplitude_uv)

    # Find all positive peaks in the waveform
    # Height parameter can be adjusted based on expected signal-to-noise ratio
    peaks, peak_properties = find_peaks(amplitude_uv, height=0)

    if len(peaks) == 0:
        return "Research Log: No positive peaks detected in the ABR waveform."

    # Calculate peak heights (amplitudes)
    peak_heights = peak_properties["peak_heights"]

    # Get the time points (latencies) of the peaks
    peak_latencies = time_ms[peaks]

    # Identify the P1 peak (typically the first significant peak, usually within 1-2 ms)
    # We'll look for peaks in the expected time window for P1 (typically 1-3 ms)
    p1_window_indices = np.where((peak_latencies >= 1) & (peak_latencies <= 3))[0]

    p1_note = ""
    if len(p1_window_indices) == 0:
        # If no peaks in the expected P1 window, use the first peak as P1 -- and say so: it is often the
        # stimulus artifact under 1 ms, and the log used to present it as Wave I without a word.
        p1_index = 0
        p1_note = (
            f"- NOTE: no positive peak in the 1-3 ms P1 window; the first positive peak "
            f"(at {peak_latencies[0]:.2f} ms) is reported as P1 -- check it is not the stimulus artifact\n"
        )
    else:
        # Use the highest peak in the P1 window
        p1_index = p1_window_indices[np.argmax(peak_heights[p1_window_indices])]

    p1_amplitude = peak_heights[p1_index]
    p1_latency = peak_latencies[p1_index]

    # Create research log
    log = (
        f"Research Log: ABR Waveform P1 Analysis\n"
        f"- Analyzed waveform with {len(time_ms)} data points\n"
        f"- Detected {len(peaks)} positive peaks in the waveform\n"
        f"{p1_note}"
        f"- Identified P1 peak metrics:\n"
        f"  * P1 Amplitude: {p1_amplitude:.2f} μV\n"
        f"  * P1 Latency: {p1_latency:.2f} ms\n"
        f"- Analysis complete"
    )

    return log


def analyze_ciliary_beat_frequency(video_path, roi_count=5, min_freq=2, max_freq=30, output_dir="./"):
    """Analyze ciliary beat frequency from high-speed video microscopy data using FFT analysis.

    Parameters
    ----------
    video_path : str
        Path to the high-speed video microscopy file of ciliary beating
    roi_count : int, optional
        Number of regions of interest to analyze (default: 5)
    min_freq : float, optional
        Minimum frequency to consider in Hz (default: 2 -- below the physiological CBF range; the lowest
        bins carry illumination drift and photobleaching, which a default of 0 reported as the beat)
    max_freq : float, optional
        Maximum frequency to consider in Hz (default: 30)
    output_dir : str, optional
        Directory to save output files (default: current directory)

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import cv2
    import numpy as np
    import pandas as pd
    from scipy.fftpack import fft
    from scipy.signal import detrend

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = []
    log.append(f"CILIARY BEAT FREQUENCY ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"Video file: {video_path}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Error: Could not open video file {video_path}"

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.append(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
    # The frequency axis is built from fps, so a container that reports none cannot be analysed: 1/fps
    # used to raise ZeroDivisionError (hunt 2026-09-30, uT7-immunology-22).
    if not fps > 0:
        cap.release()
        return f"Error: {video_path} reports a frame rate of {fps} fps, so beat frequencies cannot be computed."

    # Extract frames and prepare for analysis
    log.append("Extracting frames and preparing for analysis...")

    # Create regions of interest (ROIs)
    rois = []
    roi_size = min(width, height) // 10  # ROI size as 1/10 of the smallest dimension

    # Generate ROIs in a grid pattern
    rows = int(np.sqrt(roi_count))
    cols = (roi_count + rows - 1) // rows  # Ceiling division

    for i in range(rows):
        for j in range(cols):
            if len(rois) >= roi_count:
                break

            x = (j + 1) * width // (cols + 1) - roi_size // 2
            y = (i + 1) * height // (rows + 1) - roi_size // 2

            # Ensure ROI is within frame boundaries
            x = max(0, min(x, width - roi_size))
            y = max(0, min(y, height - roi_size))

            rois.append((x, y, roi_size, roi_size))

    log.append(f"Created {len(rois)} regions of interest for analysis")

    # Extract intensity time series for each ROI
    intensity_series = [[] for _ in range(len(rois))]

    # Read until the stream ends rather than trusting CAP_PROP_FRAME_COUNT, which is 0 or negative for a
    # broken index or a live stream -- no frame was read then, and the FFT of an empty series raised.
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i, (x, y, w, h) in enumerate(rois):
            roi = gray[y : y + h, x : x + w]
            intensity_series[i].append(np.mean(roi))

        frame_idx += 1

    cap.release()

    log.append(f"Processed {frame_idx} frames for time series analysis")
    if frame_idx < 4:
        return (
            "\n".join(log) + f"\nError: only {frame_idx} frame(s) could be read from {video_path}; no FFT was computed."
        )

    # Perform FFT analysis on each ROI
    frequencies = []

    for i, series in enumerate(intensity_series):
        # Convert to numpy array
        series = np.array(series)

        # Remove the DC component and any linear drift. Mean removal alone left photobleaching drift in the
        # lowest bins, and with min_freq=0 that drift was reported as a ~0.3 Hz beat (uT7-immunology-22).
        series = detrend(series, type="linear")

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(series))
        series = series * window

        # Compute FFT
        fft_result = fft(series)
        fft_magnitude = np.abs(fft_result[: len(series) // 2])

        # Compute frequency axis
        freq_axis = np.fft.fftfreq(len(series), d=1 / fps)[: len(series) // 2]

        # Find dominant frequency in the specified range (never the 0 Hz bin, which is no beat at all)
        valid_indices = (freq_axis >= min_freq) & (freq_axis <= max_freq) & (freq_axis > 0)
        valid_freq = freq_axis[valid_indices]
        valid_magnitude = fft_magnitude[valid_indices]

        if len(valid_magnitude) > 0 and np.max(valid_magnitude) > 0:
            dominant_idx = np.argmax(valid_magnitude)
            dominant_freq = valid_freq[dominant_idx]
            frequencies.append(dominant_freq)
            log.append(f"ROI #{i + 1}: Dominant frequency = {dominant_freq:.2f} Hz")

    # Calculate median frequency
    if frequencies:
        median_cbf = np.median(frequencies)
        log.append(f"\nResults: Detected {len(frequencies)} valid ciliary regions")
        log.append(f"Median ciliary beat frequency: {median_cbf:.2f} Hz")

        # Save results to CSV
        results_file = os.path.join(output_dir, "cbf_results.csv")
        df = pd.DataFrame({"ROI": range(1, len(frequencies) + 1), "Beat_Frequency_Hz": frequencies})
        df.to_csv(results_file, index=False)
        log.append(f"\nDetailed frequency data saved to: {results_file}")
    else:
        log.append("\nNo valid ciliary beat frequencies detected in the specified range")

    return "\n".join(log)


def analyze_protein_colocalization(channel1_path, channel2_path, output_dir="./output", threshold_method="otsu"):
    """Analyze colocalization between two fluorescently labeled proteins in microscopy images.

    Parameters
    ----------
    channel1_path : str
        Path to the first channel image file (fluorescent protein 1)
    channel2_path : str
        Path to the second channel image file (fluorescent protein 2)
    output_dir : str
        Directory to save output files (default: "./output")
    threshold_method : str
        Method for thresholding images ('otsu', 'li', or 'yen') (default: "otsu")

    Returns
    -------
    str
        Research log summarizing the colocalization analysis results and saved files

    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from skimage import exposure, filters, io

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    img1 = io.imread(channel1_path)
    img2 = io.imread(channel2_path)

    # Ensure images are same size
    if img1.shape != img2.shape:
        return "Error: Images must have the same dimensions"

    # Convert to grayscale if RGB
    if len(img1.shape) > 2:
        img1 = img1.mean(axis=2)
    if len(img2.shape) > 2:
        img2 = img2.mean(axis=2)

    # Normalize images to 0-1 range
    img1_norm = exposure.rescale_intensity(img1.astype(float), out_range=(0, 1))
    img2_norm = exposure.rescale_intensity(img2.astype(float), out_range=(0, 1))

    # Apply threshold to segment foreground from background
    if threshold_method == "otsu":
        thresh1 = filters.threshold_otsu(img1_norm)
        thresh2 = filters.threshold_otsu(img2_norm)
    elif threshold_method == "li":
        thresh1 = filters.threshold_li(img1_norm)
        thresh2 = filters.threshold_li(img2_norm)
    elif threshold_method == "yen":
        thresh1 = filters.threshold_yen(img1_norm)
        thresh2 = filters.threshold_yen(img2_norm)
    else:
        thresh1 = filters.threshold_otsu(img1_norm)
        thresh2 = filters.threshold_otsu(img2_norm)

    mask1 = img1_norm > thresh1
    mask2 = img2_norm > thresh2

    # Calculate Pearson's correlation coefficient (whole image)
    pearson_whole = stats.pearsonr(img1_norm.flatten(), img2_norm.flatten())[0]

    # Calculate Pearson's correlation coefficient (above threshold)
    combined_mask = np.logical_or(mask1, mask2)
    if np.sum(combined_mask) > 0:
        pearson_masked = stats.pearsonr(img1_norm[combined_mask].flatten(), img2_norm[combined_mask].flatten())[0]
    else:
        pearson_masked = 0

    # Calculate Mander's Overlap Coefficient (MOC)
    numerator = np.sum(img1_norm * img2_norm)
    denominator = np.sqrt(np.sum(img1_norm**2) * np.sum(img2_norm**2))
    moc = numerator / denominator if denominator > 0 else 0

    # Calculate Mander's Colocalization Coefficients (MCC)
    # M1: Fraction of channel1 overlapping with channel2
    # M2: Fraction of channel2 overlapping with channel1
    m1 = np.sum(img1_norm * mask2) / np.sum(img1_norm) if np.sum(img1_norm) > 0 else 0
    m2 = np.sum(img2_norm * mask1) / np.sum(img2_norm) if np.sum(img2_norm) > 0 else 0

    # Create visualization
    plt.figure(figsize=(10, 8))

    # Plot original images and overlay
    plt.subplot(2, 2, 1)
    plt.imshow(img1_norm, cmap="Greens")
    plt.title("Channel 1")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img2_norm, cmap="Reds")
    plt.title("Channel 2")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    overlay = np.zeros((img1_norm.shape[0], img1_norm.shape[1], 3))
    overlay[:, :, 1] = img1_norm  # Green channel
    overlay[:, :, 0] = img2_norm  # Red channel
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    # Plot scatter plot of pixel intensities
    plt.subplot(2, 2, 4)
    plt.hexbin(img1_norm.flatten(), img2_norm.flatten(), gridsize=50, cmap="viridis", mincnt=1)
    plt.xlabel("Channel 1 Intensity")
    plt.ylabel("Channel 2 Intensity")
    plt.title("Intensity Correlation")

    # Save visualization
    viz_filename = os.path.join(output_dir, "colocalization_visualization.png")
    plt.tight_layout()
    plt.savefig(viz_filename)
    plt.close()

    # Create research log
    log = f"""Colocalization Analysis Research Log:

1. Analyzed images:
   - Channel 1: {channel1_path}
   - Channel 2: {channel2_path}

2. Thresholding method: {threshold_method}
   - Channel 1 threshold: {thresh1:.4f}
   - Channel 2 threshold: {thresh2:.4f}

3. Colocalization metrics:
   - Pearson's correlation coefficient (whole image): {pearson_whole:.4f}
   - Pearson's correlation coefficient (above threshold): {pearson_masked:.4f}
   - Mander's Overlap Coefficient (MOC): {moc:.4f}
   - Mander's Colocalization Coefficient M1: {m1:.4f}
   - Mander's Colocalization Coefficient M2: {m2:.4f}

4. Visualization saved to: {viz_filename}

Interpretation:
- Pearson's coefficient ranges from -1 to 1, with values closer to 1 indicating stronger colocalization
- Mander's coefficients range from 0 to 1, with values closer to 1 indicating higher overlap
- M1 represents the fraction of channel 1 overlapping with channel 2
- M2 represents the fraction of channel 2 overlapping with channel 1
"""

    return log


def perform_cosinor_analysis(time_data, physiological_data, period=24.0):
    """Performs cosinor analysis on physiological time series data to characterize circadian rhythms.

    Parameters
    ----------
    time_data : array-like
        Time points of the measurements in hours
    physiological_data : array-like
        Physiological measurements corresponding to each time point
    period : float, default=24.0
        Period of the rhythm in hours, default is 24 hours for circadian rhythms

    Returns
    -------
    str
        A research log summarizing the cosinor analysis process and results

    """
    from datetime import datetime

    import numpy as np

    # Fit the cosinor model  y = mesor + amplitude * cos(2*pi*t/period - acrophase)  in its linear form
    # y = mesor + beta*cos(w t) + gamma*sin(w t) by least squares; then amplitude = hypot(beta, gamma) and
    # acrophase = atan2(gamma, beta). The nonlinear fit this replaces started at acrophase 0 and, for a
    # rhythm peaking near mid-period, flipped the amplitude's sign instead of moving the phase -- so it
    # reported the trough as the peak, 12 h off (hunt 2026-09-30, uT7-immunology-6).
    try:
        time_data = np.asarray(time_data, dtype=float)
        physiological_data = np.asarray(physiological_data, dtype=float)
        if time_data.shape != physiological_data.shape or time_data.ndim != 1:
            raise ValueError(
                f"time_data and physiological_data must be 1-D and the same length "
                f"(got {time_data.shape} and {physiological_data.shape})"
            )
        n = len(time_data)
        if n < 3:
            raise ValueError(f"need at least 3 time points to fit mesor, amplitude and acrophase (got {n})")

        omega_t = 2 * np.pi * time_data / period
        design = np.column_stack([np.ones(n), np.cos(omega_t), np.sin(omega_t)])
        coefficients = np.linalg.lstsq(design, physiological_data, rcond=None)[0]
        mesor, beta, gamma = coefficients

        # Extract parameters
        amplitude = float(np.hypot(beta, gamma))
        acrophase = float(np.arctan2(gamma, beta) % (2 * np.pi))

        # Convert acrophase to hours (peak time)
        acrophase_hours = (acrophase * period) / (2 * np.pi)
        acrophase_hours = acrophase_hours % period  # Ensure it's within the period

        # Calculate model fit
        predicted_values = design @ coefficients

        # Calculate goodness of fit metrics
        residuals = physiological_data - predicted_values
        ss_total = np.sum((physiological_data - np.mean(physiological_data)) ** 2)
        ss_residual = np.sum(residuals**2)
        r_squared = 1 - (ss_residual / ss_total)

        # Standard errors: the coefficient covariance (residual variance scaled, as curve_fit reported it),
        # carried to amplitude and acrophase by the delta method
        std_errors = [float("nan")] * 3
        if n > 3:
            covariance = ss_residual / (n - 3) * np.linalg.pinv(design.T @ design)
            std_errors[0] = float(np.sqrt(covariance[0, 0]))
            if amplitude > 0:
                grad_amplitude = np.array([beta, gamma]) / amplitude
                grad_acrophase = np.array([-gamma, beta]) / amplitude**2
                harmonic = covariance[1:, 1:]
                std_errors[1] = float(np.sqrt(grad_amplitude @ harmonic @ grad_amplitude))
                std_errors[2] = float(np.sqrt(grad_acrophase @ harmonic @ grad_acrophase))

        # Create results summary
        results = {
            "Mesor": f"{mesor:.4f} ± {std_errors[0]:.4f}",
            "Amplitude": f"{amplitude:.4f} ± {std_errors[1]:.4f}",
            "Acrophase (radians)": f"{acrophase:.4f} ± {std_errors[2]:.4f}",
            "Peak Time (hours)": f"{acrophase_hours:.2f}",
            "R-squared": f"{r_squared:.4f}",
        }

        # Create a research log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = f"Cosinor Analysis Research Log - {timestamp}\n"
        log += "=" * 50 + "\n\n"
        log += "ANALYSIS SUMMARY:\n"
        log += f"Analyzed {len(time_data)} data points over {max(time_data) - min(time_data):.1f} hours\n"
        log += f"Fitted a cosine curve with period = {period} hours\n\n"

        log += "CIRCADIAN RHYTHM PARAMETERS:\n"
        for param, value in results.items():
            log += f"- {param}: {value}\n"

        log += "\nINTERPRETATION:\n"
        log += f"- The rhythm-adjusted mean (mesor) is {mesor:.4f}\n"
        log += f"- The rhythm shows a peak-to-trough variation of {2 * amplitude:.4f}\n"
        log += f"- The peak of the rhythm occurs at {acrophase_hours:.2f} hours\n"
        log += f"- The model explains {r_squared * 100:.1f}% of the variation in the data\n"

        if r_squared < 0.3:
            log += "\nNOTE: The low R-squared suggests a weak circadian pattern in this data.\n"

        return log

    except Exception as e:
        return f"Error in cosinor analysis: {str(e)}"


def calculate_brain_adc_map(dwi_file_path, b_values, output_path="adc_map.nii.gz", mask_file_path=None):
    """Calculate Apparent Diffusion Coefficient (ADC) map from diffusion-weighted MRI data.

    This function fits the DW-MRI data to the monoexponential diffusion model:
    S = S0 * exp(-b * ADC), where S is the signal intensity, S0 is the signal without
    diffusion weighting, b is the b-value, and ADC is the apparent diffusion coefficient.

    Parameters
    ----------
    dwi_file_path : str
        Path to the 4D NIfTI file containing diffusion-weighted MRI data,
        where the 4th dimension corresponds to different b-values.
    b_values : list or numpy.ndarray
        List of b-values corresponding to each volume in the 4D DWI data.
    output_path : str, optional
        Path where the output ADC map will be saved (default: "adc_map.nii.gz").
    mask_file_path : str, optional
        Path to a binary mask file to limit ADC calculation to brain regions (default: None).

    Returns
    -------
    str
        Research log summarizing the ADC mapping process and results.

    """
    import os

    import nibabel as nib
    import numpy as np
    from scipy.optimize import curve_fit

    # Start research log
    research_log = "Brain Water Diffusion Mapping Research Log\n"
    research_log += "=" * 50 + "\n\n"

    # Step 1: Load DWI data
    research_log += "Step 1: Loading diffusion-weighted MRI data\n"
    dwi_img = nib.load(dwi_file_path)
    dwi_data = dwi_img.get_fdata()
    research_log += f"- Loaded DWI data with shape: {dwi_data.shape}\n"
    research_log += f"- B-values: {b_values}\n\n"

    # One b-value per volume, or every voxel fit fails on a shape mismatch -- which the per-voxel except
    # below swallowed, writing an all-zero map reported as a success (hunt 2026-09-30, uT7-immunology-13).
    if dwi_data.ndim != 4 or len(b_values) != dwi_data.shape[3]:
        n_volumes = dwi_data.shape[3] if dwi_data.ndim == 4 else "no 4th dimension"
        return research_log + (
            f"Error: b_values has {len(b_values)} entries but the DWI data has {n_volumes} volumes; pass one "
            "b-value per volume (repeat a b-value for each diffusion direction acquired at it). "
            "No ADC map was written.\n"
        )

    # Step 2: Load mask if provided
    if mask_file_path:
        research_log += "Step 2: Loading brain mask\n"
        mask_img = nib.load(mask_file_path)
        mask_data = mask_img.get_fdata() > 0  # Convert to boolean
        research_log += f"- Loaded mask with {np.sum(mask_data)} voxels to process\n\n"
    else:
        research_log += "Step 2: No mask provided, processing all voxels\n"
        # Create a mask where signal in the baseline image is above a threshold
        baseline_idx = np.argmin(b_values)  # Typically b=0 image
        mask_data = dwi_data[:, :, :, baseline_idx] > np.mean(dwi_data[:, :, :, baseline_idx]) * 0.1
        research_log += f"- Created automatic mask with {np.sum(mask_data)} voxels to process\n\n"

    # Step 3: Define the monoexponential diffusion model
    research_log += "Step 3: Fitting data to monoexponential diffusion model\n"
    research_log += "- Using model: S = S0 * exp(-b * ADC)\n"

    def mono_exp_model(b, S0, ADC):
        return S0 * np.exp(-b * ADC)

    # Step 4: Calculate ADC map
    shape = dwi_data.shape[:3]
    adc_map = np.zeros(shape)
    S0_map = np.zeros(shape)

    # Convert b_values to numpy array if it's not already
    b_values = np.array(b_values)

    # Process only voxels within the mask
    total_voxels = np.sum(mask_data)
    processed_voxels = 0

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if mask_data[x, y, z]:
                    signal = dwi_data[x, y, z, :]

                    # Skip voxels with zero or negative values
                    if np.any(signal <= 0):
                        continue

                    try:
                        # Fit the monoexponential model
                        params, _ = curve_fit(
                            mono_exp_model,
                            b_values,
                            signal,
                            p0=[signal[0], 0.001],  # Initial guess
                            bounds=([0, 0], [np.inf, 0.01]),
                        )
                        S0_map[x, y, z] = params[0]
                        adc_map[x, y, z] = params[1]
                        processed_voxels += 1
                    except Exception:
                        # If fitting fails, set ADC to 0
                        adc_map[x, y, z] = 0

    research_log += f"- Successfully processed {processed_voxels} out of {total_voxels} voxels\n\n"
    if processed_voxels == 0:
        return research_log + (
            "Error: no voxel could be fitted to the monoexponential model (voxels with a zero or negative "
            "signal are skipped, and every fit that was attempted failed). No ADC map was written.\n"
        )

    # Step 5: Save ADC map
    research_log += "Step 5: Saving ADC map\n"
    # Convert ADC from mm²/s to µm²/ms for conventional display
    adc_map_save = adc_map * 1000  # Convert to µm²/ms
    adc_img = nib.Nifti1Image(adc_map_save, dwi_img.affine, dwi_img.header)
    nib.save(adc_img, output_path)
    research_log += f"- ADC map saved to: {os.path.abspath(output_path)}\n"

    # Step 6: Calculate summary statistics
    valid_adcs = adc_map[mask_data & (adc_map > 0)]
    if len(valid_adcs) > 0:
        research_log += "\nStep 6: ADC Statistics (in µm²/ms):\n"
        research_log += f"- Mean ADC: {np.mean(valid_adcs) * 1000:.4f}\n"
        research_log += f"- Median ADC: {np.median(valid_adcs) * 1000:.4f}\n"
        research_log += f"- Min ADC: {np.min(valid_adcs) * 1000:.4f}\n"
        research_log += f"- Max ADC: {np.max(valid_adcs) * 1000:.4f}\n"
        research_log += f"- Std Dev ADC: {np.std(valid_adcs) * 1000:.4f}\n"

    research_log += "\nBrain water diffusion mapping completed successfully.\n"

    return research_log


def analyze_endolysosomal_calcium_dynamics(
    time_points,
    luminescence_values,
    treatment_time=None,
    cell_type="",
    treatment_name="",
    output_file="calcium_analysis_results.txt",
):
    """Analyze calcium dynamics in endo-lysosomal compartments using ELGA/ELGA1 probe data.

    Parameters
    ----------
    time_points : numpy.ndarray or list
        Time points of the measurements in seconds
    luminescence_values : numpy.ndarray or list
        Luminescence intensity values from ELGA/ELGA1 probes corresponding to Ca2+ levels
    treatment_time : float, optional
        Time point (in seconds) when treatment/stimulus was applied
    cell_type : str, optional
        Type of cells used in the experiment
    treatment_name : str, optional
        Name of the treatment or stimulus applied
    output_file : str, optional
        Name of the file to save detailed analysis results

    Returns
    -------
    str
        A research log summarizing the calcium dynamics analysis and key findings

    """
    import os

    import numpy as np
    from scipy import signal

    # Convert inputs to numpy arrays if they aren't already
    time_points = np.array(time_points)
    luminescence_values = np.array(luminescence_values)

    # Calculate baseline Ca2+ levels (average of pre-treatment values or first 10% of data). Under 10 points
    # the 10% window was empty, and a treatment_time at or before the first sample emptied the
    # pre-treatment one: the baseline was NaN, every normalised value NaN, and a 3x rise was reported as
    # "No significant Ca2+ peaks" (hunt 2026-09-30, uT7-immunology-27).
    if treatment_time is not None:
        baseline_indices = time_points < treatment_time
        if not np.any(baseline_indices):
            return (
                f"Error: no measurement precedes treatment_time={treatment_time} (first time point: "
                f"{time_points[0] if len(time_points) else 'none'}), so there is no pre-treatment baseline."
            )
        baseline = np.mean(luminescence_values[baseline_indices])
        baseline_std = np.std(luminescence_values[baseline_indices])
    else:
        baseline_idx = max(1, int(len(luminescence_values) * 0.1))  # First 10% of data, at least one point
        baseline = np.mean(luminescence_values[:baseline_idx])
        baseline_std = np.std(luminescence_values[:baseline_idx])
    if not np.isfinite(baseline) or baseline == 0:
        return f"Error: the baseline luminescence is {baseline}, so the trace cannot be normalised to it."

    # Normalize data to baseline
    normalized_values = luminescence_values / baseline

    # Find peaks in Ca2+ signal
    peaks, properties = signal.find_peaks(normalized_values, height=1.1, distance=5)

    # Calculate key metrics
    if len(peaks) > 0:
        peak_heights = properties["peak_heights"]
        max_peak_idx = peaks[np.argmax(peak_heights)]
        max_peak_time = time_points[max_peak_idx]
        max_peak_value = normalized_values[max_peak_idx]

        # Calculate response kinetics if treatment_time is provided
        if treatment_time is not None and max_peak_time > treatment_time:
            response_time = max_peak_time - treatment_time
        else:
            response_time = None

        # Calculate area under the curve (Ca2+ load)
        auc = np.trapezoid(normalized_values - 1, time_points)

        # Calculate decay time (time to return to 50% of peak)
        if max_peak_idx < len(normalized_values) - 1:
            post_peak = normalized_values[max_peak_idx:]
            post_peak_times = time_points[max_peak_idx:]
            half_peak_value = 1 + (max_peak_value - 1) / 2

            # Find where signal drops below half-peak
            below_half_indices = np.where(post_peak < half_peak_value)[0]
            if len(below_half_indices) > 0:
                half_decay_idx = below_half_indices[0]
                decay_time = post_peak_times[half_decay_idx] - max_peak_time
            else:
                decay_time = None
        else:
            decay_time = None
    else:
        max_peak_value = None
        max_peak_time = None
        response_time = None
        auc = np.trapezoid(normalized_values - 1, time_points)
        decay_time = None

    # Calculate coefficient of variation to measure Ca2+ fluctuations
    cv = np.std(normalized_values) / np.mean(normalized_values)

    # Save detailed results to file
    with open(output_file, "w") as f:
        f.write("Endo-lysosomal Ca2+ Dynamics Analysis Results\n")
        f.write("===========================================\n\n")
        f.write(f"Cell Type: {cell_type}\n")
        f.write(f"Treatment: {treatment_name}\n\n")
        f.write(f"Baseline Ca2+ level: {baseline:.2f} ± {baseline_std:.2f}\n")
        f.write(f"Number of detected Ca2+ peaks: {len(peaks)}\n")

        if max_peak_value is not None:
            f.write(f"Maximum Ca2+ peak: {max_peak_value:.2f} (normalized to baseline)\n")
            f.write(f"Time of maximum peak: {max_peak_time:.2f} seconds\n")

        if response_time is not None:
            f.write(f"Response time after treatment: {response_time:.2f} seconds\n")

        if decay_time is not None:
            f.write(f"Half-decay time: {decay_time:.2f} seconds\n")

        f.write(f"Area under curve (Ca2+ load): {auc:.2f}\n")
        f.write(f"Coefficient of variation: {cv:.4f}\n\n")

        if len(peaks) > 0:
            f.write("All detected peaks (time, normalized intensity):\n")
            for i, peak_idx in enumerate(peaks):
                f.write(f"  Peak {i + 1}: {time_points[peak_idx]:.2f}s, {normalized_values[peak_idx]:.2f}\n")

    # Generate research log
    log = f"""
Research Log: Endo-lysosomal Ca2+ Dynamics Analysis using ELGA/ELGA1 Probes

SUMMARY:
Analyzed calcium dynamics in the endo-lysosomal compartment of {cell_type} cells using ELGA/ELGA1 probes.
{f"Cells were treated with {treatment_name} at {treatment_time} seconds." if treatment_time is not None else ""}

METHODS:
- Measured luminescence values from ELGA/ELGA1 probes over {len(time_points)} time points
- Calculated baseline Ca2+ levels and normalized signals
- Identified Ca2+ peaks and analyzed response kinetics
- Quantified Ca2+ load using area under curve analysis

RESULTS:
- Baseline Ca2+ level: {baseline:.2f} ± {baseline_std:.2f} (arbitrary units)
- Detected {len(peaks)} Ca2+ peaks in the endo-lysosomal compartment
{f"- Maximum Ca2+ response: {max_peak_value:.2f}x baseline" if max_peak_value is not None else "- No significant Ca2+ peaks detected"}
{f"- Response time: {response_time:.2f} seconds after treatment" if response_time is not None else ""}
{f"- Half-decay time: {decay_time:.2f} seconds" if decay_time is not None else ""}
- Total Ca2+ load (AUC): {auc:.2f}
- Ca2+ signal variability (CV): {cv:.4f}

Detailed numerical results saved to: {os.path.abspath(output_file)}
"""

    return log


def analyze_fatty_acid_composition_by_gc(gc_data_file, tissue_type, sample_id, output_directory="./results"):
    """Analyzes fatty acid composition in tissue samples using gas chromatography data.

    Parameters
    ----------
    gc_data_file : str
        Path to the CSV file containing gas chromatography data with columns 'retention_time' and 'peak_area'.
    tissue_type : str
        Type of tissue sample (e.g., liver, kidney, heart, muscle, adipose).
    sample_id : str
        Identifier for the sample being analyzed.
    output_directory : str, optional
        Directory where result files will be saved (default: "./results").

    Returns
    -------
    str
        Research log summarizing the analysis steps and results.

    """
    import os
    from datetime import datetime

    import pandas as pd

    os.makedirs(output_directory, exist_ok=True)

    log = "FATTY ACID COMPOSITION ANALYSIS LOG\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Sample ID: {sample_id}\n"
    log += f"Tissue Type: {tissue_type}\n\n"

    log += "Step 1: Loading gas chromatography data\n"
    try:
        gc_data = pd.read_csv(gc_data_file)
        expected_cols = {"retention_time", "peak_area"}
        if not expected_cols.issubset(gc_data.columns):
            missing_cols = expected_cols - set(gc_data.columns)
            log += f"  - Error: Missing columns: {', '.join(missing_cols)}\n"
            return log
        log += f"  - Successfully loaded data from {gc_data_file}\n"
        log += f"  - Data contains {len(gc_data)} data points\n\n"
    except Exception as e:
        log += f"  - Error loading data: {str(e)}\n"
        return log

    log += "Step 2: Identifying fatty acid peaks\n"

    fatty_acid_standards = {
        (2.1, 2.3): "Myristic acid (C14:0)",
        (2.8, 3.0): "Palmitic acid (C16:0)",
        (3.1, 3.3): "Palmitoleic acid (C16:1)",
        (3.7, 3.9): "Stearic acid (C18:0)",
        (4.0, 4.2): "Oleic acid (C18:1)",
        (4.4, 4.6): "Linoleic acid (C18:2)",
        (4.7, 4.9): "α-Linolenic acid (C18:3)",
        (5.1, 5.3): "Arachidonic acid (C20:4)",
        (5.5, 5.7): "EPA (C20:5)",
        (6.3, 6.5): "DHA (C22:6)",
        (5.8, 6.0): "t10c12-CLA",
    }

    total_area = gc_data["peak_area"].sum()
    if total_area == 0:
        log += "  - Error: Total peak area is zero, cannot calculate percentages.\n"
        return log

    identified_peaks = {}
    for (rt_min, rt_max), fatty_acid in fatty_acid_standards.items():
        peaks_in_range = gc_data[(gc_data["retention_time"] >= rt_min) & (gc_data["retention_time"] <= rt_max)]
        area = peaks_in_range["peak_area"].sum()
        percentage = (area / total_area) * 100 if area > 0 else 0.0
        identified_peaks[fatty_acid] = {"area": area, "percentage": percentage}
        log += f"  - {fatty_acid}: {percentage:.2f}% ({'Detected' if area > 0 else 'Not Detected'})\n"

    detected_count = sum(1 for fa in identified_peaks.values() if fa["area"] > 0)
    log += f"  - Total detected fatty acids: {detected_count}/{len(fatty_acid_standards)}\n\n"

    log += "Step 3: Generating fatty acid composition results\n"
    results = (
        pd.DataFrame(
            {
                "Fatty Acid": list(identified_peaks),
                "Peak Area": [identified_peaks[fa]["area"] for fa in identified_peaks],
                "Percentage (%)": [identified_peaks[fa]["percentage"] for fa in identified_peaks],
            }
        )
        .sort_values("Percentage (%)", ascending=False)
        .reset_index(drop=True)
    )

    output_file = os.path.join(
        output_directory,
        f"{_safe_filename_part(sample_id)}_{_safe_filename_part(tissue_type)}_fatty_acid_composition.csv",
    )
    results.to_csv(output_file, index=False)
    log += f"  - Results saved to: {output_file}\n\n"

    log += "Step 4: Summary of fatty acid composition\n"
    detected_results = results[results["Peak Area"] > 0]
    if not detected_results.empty:
        top_fatty_acids = detected_results.head(3)
        log += "  - Most abundant fatty acids:\n"
        for _, row in top_fatty_acids.iterrows():
            log += f"    * {row['Fatty Acid']}: {row['Percentage (%)']:.2f}%\n"

        saturated = [
            "Myristic acid (C14:0)",
            "Palmitic acid (C16:0)",
            "Stearic acid (C18:0)",
        ]
        unsaturated = [fa for fa in identified_peaks if fa not in saturated]

        saturated_total = sum(identified_peaks[fa]["percentage"] for fa in saturated)
        unsaturated_total = sum(identified_peaks[fa]["percentage"] for fa in unsaturated)

        if unsaturated_total > 0:
            sat_unsat_ratio = saturated_total / unsaturated_total
            log += f"  - Saturated to unsaturated fatty acid ratio: {sat_unsat_ratio:.2f}\n"
        else:
            log += "  - Unsaturated fatty acids not detected; cannot calculate ratio.\n"
    else:
        log += "  - No fatty acids detected; please verify input data.\n"

    log += "\nAnalysis completed successfully."

    return log


def analyze_hemodynamic_data(pressure_data, sampling_rate, output_file="hemodynamic_results.csv"):
    """Analyzes raw blood pressure data to calculate key hemodynamic parameters.

    Parameters
    ----------
    pressure_data : numpy.ndarray
        Raw blood pressure measurements in mmHg
    sampling_rate : float
        Data acquisition rate in Hz (samples per second)
    output_file : str, optional
        Filename to save the calculated parameters, default is "hemodynamic_results.csv"

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os

    import numpy as np
    import pandas as pd
    from scipy import signal

    pressure_data = np.asarray(pressure_data, dtype=float)

    # Step 1: Preprocess the signal
    # The band-pass below only LOCATES the beats. Its 0.5 Hz high-pass removes the mean pressure, so
    # every amplitude is read from the raw trace; reading SBP/DBP off the filtered, zero-centred copy
    # reported a 120/80 mmHg waveform as 20/-20 mmHg (hunt 2026-09-30, uT7-immunology-5).
    nyquist_freq = sampling_rate / 2
    if not nyquist_freq > 10:
        return (
            f"Error: sampling_rate={sampling_rate} Hz is too low. Beats are located on a 0.5-10 Hz band-passed "
            "copy of the trace, which needs a sampling_rate above 20 Hz (the 10 Hz edge must sit below Nyquist)."
        )
    low_cutoff = 0.5 / nyquist_freq  # 0.5 Hz (to preserve slow components)
    high_cutoff = 10 / nyquist_freq  # 10 Hz (to preserve fast components but remove noise)
    b, a = signal.butter(2, [low_cutoff, high_cutoff], btype="band")
    filtered_data = signal.filtfilt(b, a, pressure_data)

    # Step 2: Find peaks (systolic) and troughs (diastolic)
    # For systolic peaks
    peaks, _ = signal.find_peaks(filtered_data, distance=sampling_rate * 0.5)  # Minimum distance between peaks
    if len(peaks) < 2:
        return (
            f"Error: detected {len(peaks)} systolic peak(s) in {len(pressure_data)} samples; at least two beats "
            "are needed to measure blood pressure and heart rate."
        )

    # Systolic value of each beat: the raw maximum within 0.1 s of the filtered peak
    half_window = max(1, int(round(0.1 * sampling_rate)))
    sbp_values = np.array([pressure_data[max(0, p - half_window) : p + half_window + 1].max() for p in peaks])

    # For diastolic troughs (raw minima between systolic peaks)
    valleys = []
    for i in range(len(peaks) - 1):
        segment = pressure_data[peaks[i] : peaks[i + 1]]
        valley_idx = np.argmin(segment) + peaks[i]
        valleys.append(valley_idx)

    # Step 3: Calculate hemodynamic parameters
    # Systolic Blood Pressure (SBP)
    sbp = np.mean(sbp_values)

    # Diastolic Blood Pressure (DBP)
    dbp_values = pressure_data[valleys]
    dbp = np.mean(dbp_values)

    # Mean Arterial Pressure (MAP): the time-average of the raw pressure over whole cardiac cycles
    map_value = np.mean(pressure_data[peaks[0] : peaks[-1]])

    # Heart Rate (HR)
    # Calculate average time between peaks and convert to beats per minute
    peak_intervals = np.diff(peaks) / sampling_rate  # Convert to seconds
    heart_rate = 60 / np.mean(peak_intervals)  # Convert to beats per minute

    # Step 4: Save results to CSV
    results_df = pd.DataFrame(
        {
            "Parameter": ["SBP", "DBP", "MAP", "HR"],
            "Value": [sbp, dbp, map_value, heart_rate],
            "Unit": ["mmHg", "mmHg", "mmHg", "bpm"],
        }
    )
    results_df.to_csv(output_file, index=False)

    # Generate research log
    log = f"""Hemodynamic Data Analysis Log:
1. Located beats on a band-pass filtered copy (0.5-10 Hz); pressures were read from the raw signal
2. Detected {len(peaks)} systolic peaks and {len(valleys)} diastolic troughs
3. Calculated hemodynamic parameters:
   - Systolic Blood Pressure (SBP): {sbp:.2f} mmHg
   - Diastolic Blood Pressure (DBP): {dbp:.2f} mmHg
   - Mean Arterial Pressure (MAP): {map_value:.2f} mmHg
   - Heart Rate (HR): {heart_rate:.2f} bpm
4. Results saved to {os.path.abspath(output_file)}
"""

    return log


def simulate_thyroid_hormone_pharmacokinetics(
    parameters, initial_conditions, time_span=(0, 24), time_points=100, output_dir="."
):
    """Simulates the transport and binding of thyroid hormones across different tissue compartments
    using an ODE-based pharmacokinetic model.

    Parameters
    ----------
    parameters : dict
        Dictionary containing model parameters:
        - transport_rates : dict of rate constants for hormone transport between compartments
        - binding_constants : dict of association/dissociation constants for hormone-protein binding
        - metabolism_rates : dict of rate constants for hormone metabolism in different tissues
        - volumes : dict of compartment volumes

    initial_conditions : dict
        Dictionary of initial concentrations for all molecular species in each compartment

    time_span : tuple, optional
        Start and end time for simulation in hours (default: (0, 24))

    time_points : int, optional
        Number of time points to output (default: 100)

    output_dir : str, optional
        Directory to save the concentration profiles CSV (default: current directory)

    Returns
    -------
    str
        Research log summarizing the simulation and results

    """
    import os

    import numpy as np
    import pandas as pd
    from scipy.integrate import solve_ivp

    # Extract parameters
    transport_rates = parameters.get("transport_rates", {})
    binding_constants = parameters.get("binding_constants", {})
    metabolism_rates = parameters.get("metabolism_rates", {})
    volumes = parameters.get("volumes", {})

    # Convert initial conditions dictionary to array for solver
    species_names = list(initial_conditions.keys())
    y0 = [initial_conditions[name] for name in species_names]

    # Define the ODE system
    def pk_ode_system(t, y):
        # Initialize derivatives
        dydt = np.zeros_like(y)

        # Map array indices back to named species for readability
        species = {name: y[i] for i, name in enumerate(species_names)}

        # Example: For each compartment, calculate concentration changes

        # Blood compartment (example)
        if "T4_blood_free" in species:
            blood_idx = species_names.index("T4_blood_free")
            # Transport from blood to tissues
            for tissue in ["liver", "thyroid", "kidney"]:
                tissue_key = f"T4_{tissue}_free"
                if tissue_key in species:
                    tissue_idx = species_names.index(tissue_key)
                    rate_key = f"blood_to_{tissue}"
                    if rate_key in transport_rates:
                        # Transport out of blood
                        dydt[blood_idx] -= (
                            transport_rates[rate_key] * species["T4_blood_free"] / volumes.get("blood", 1)
                        )
                        # Transport into tissue
                        dydt[tissue_idx] += (
                            transport_rates[rate_key] * species["T4_blood_free"] / volumes.get(tissue, 1)
                        )

        # Protein binding in blood (example)
        if "T4_blood_free" in species and "TBG_blood" in species and "T4_TBG_complex" in species:
            free_idx = species_names.index("T4_blood_free")
            protein_idx = species_names.index("TBG_blood")
            complex_idx = species_names.index("T4_TBG_complex")

            # Association
            if "k_on_T4_TBG" in binding_constants:
                association = binding_constants["k_on_T4_TBG"] * species["T4_blood_free"] * species["TBG_blood"]
                dydt[free_idx] -= association
                dydt[protein_idx] -= association
                dydt[complex_idx] += association

            # Dissociation
            if "k_off_T4_TBG" in binding_constants:
                dissociation = binding_constants["k_off_T4_TBG"] * species["T4_TBG_complex"]
                dydt[free_idx] += dissociation
                dydt[protein_idx] += dissociation
                dydt[complex_idx] -= dissociation

        # Metabolism (example for liver)
        if "T4_liver_free" in species and "T3_liver_free" in species:
            t4_idx = species_names.index("T4_liver_free")
            t3_idx = species_names.index("T3_liver_free")

            if "T4_to_T3_liver" in metabolism_rates:
                conversion = metabolism_rates["T4_to_T3_liver"] * species["T4_liver_free"]
                dydt[t4_idx] -= conversion
                dydt[t3_idx] += conversion

        return dydt

    # Solve the ODE system
    t_eval = np.linspace(time_span[0], time_span[1], time_points)
    solution = solve_ivp(
        pk_ode_system,
        time_span,
        y0,
        method="BDF",  # Equivalent to MATLAB's ode15s
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
    )

    # Process results
    if solution.success:
        # Create dataframe with results
        results_df = pd.DataFrame(solution.y.T, columns=species_names)
        results_df.insert(0, "Time", solution.t)

        # Save to CSV
        # Absolute, so the path this log reports back is one the caller can actually find. output_dir lets
        # the caller put it in the chat's output directory instead of the REPL's working directory, where
        # every chat of an account overwrote the one fixed name (hunt 2026-09-30, uT7-immunology-21).
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.abspath(os.path.join(output_dir, "thyroid_hormone_pk_simulation_results.csv"))
        results_df.to_csv(output_file, index=False)

        # Generate research log
        log = "## Thyroid Hormone Pharmacokinetic Simulation Results\n\n"
        log += f"- Simulation time span: {time_span[0]} to {time_span[1]} hours\n"
        log += f"- Number of molecular species modeled: {len(species_names)}\n"
        log += (
            f"- Compartments included: {', '.join({name.split('_')[1] for name in species_names if '_' in name})}\n\n"
        )

        log += "### Key Observations:\n"
        # Find peak concentrations and times
        for species in species_names:
            if "free" in species:  # Focus on free hormone concentrations
                idx = species_names.index(species)
                peak_conc = np.max(solution.y[idx])
                peak_time = solution.t[np.argmax(solution.y[idx])]
                final_conc = solution.y[idx][-1]

                log += f"- {species}: Peak concentration of {peak_conc:.4g} at {peak_time:.2f} hours, "
                log += f"final concentration of {final_conc:.4g}\n"

        log += f"\nDetailed concentration profiles saved to: {output_file}\n"

        return log
    else:
        return f"Simulation failed: {solution.message}"


def quantify_amyloid_beta_plaques(
    image_path,
    output_dir="./results",
    threshold_method="otsu",
    min_plaque_size=50,
    manual_threshold=127,
):
    import os
    from datetime import datetime

    import pandas as pd
    from skimage import color, filters, io, measure, morphology
    from skimage.segmentation import clear_border
    from skimage.util import img_as_ubyte

    os.makedirs(output_dir, exist_ok=True)
    original_image = io.imread(image_path)

    if len(original_image.shape) > 2 and original_image.shape[2] > 1:
        # PNG exports often carry an alpha channel, which rgb2gray rejects (hunt 2026-09-30, uT7-immunology-23).
        if original_image.shape[2] == 4:
            original_image = color.rgba2rgb(original_image)
        gray_image = color.rgb2gray(original_image)
    else:
        gray_image = original_image

    gray_image = img_as_ubyte(gray_image)
    # preserve_range keeps the smoothed image on the same 0-255 scale as manual_threshold (default 127);
    # without it gaussian returned [0, 1] and a manual threshold could never be exceeded, so every manual
    # call reported "No plaques detected" (hunt 2026-09-30, uT7-immunology-11).
    smoothed_image = filters.gaussian(gray_image, sigma=1, preserve_range=True)

    if smoothed_image.shape[0] < 10 or smoothed_image.shape[1] < 10:
        raise ValueError("Image too small for reliable plaque detection.")

    if threshold_method == "otsu":
        threshold_value = filters.threshold_otsu(smoothed_image)
        binary_image = smoothed_image > threshold_value
    elif threshold_method == "adaptive":
        block_size = 35 if min(smoothed_image.shape) > 35 else min(smoothed_image.shape) // 2 * 2 + 1
        adaptive_thresh = filters.threshold_local(smoothed_image, block_size=block_size)
        binary_image = smoothed_image > adaptive_thresh
    else:
        threshold_value = manual_threshold
        binary_image = smoothed_image > threshold_value

    cleaned_binary = morphology.remove_small_objects(binary_image, min_size=min_plaque_size)
    labeled_image = measure.label(cleaned_binary)
    labeled_image = clear_border(labeled_image)

    regions = measure.regionprops(labeled_image, intensity_image=gray_image)
    plaque_data = []

    for region in regions:
        plaque_data.append(
            {
                "Area": region.area,
                "Perimeter": region.perimeter,
                "Mean_Intensity": region.mean_intensity,
                "Eccentricity": region.eccentricity,
            }
        )

    total_plaque_area = sum(r["Area"] for r in plaque_data)
    total_image_area = gray_image.size
    area_fraction = (total_plaque_area / total_image_area) * 100

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_dir}/plaque_analysis_{timestamp}.csv"
    segmented_image_filename = f"{output_dir}/segmented_plaques_{timestamp}.png"

    if plaque_data:
        df = pd.DataFrame(plaque_data)
        df.to_csv(csv_filename, index=False)

        colored_labels = color.label2rgb(labeled_image, image=gray_image, bg_label=0, kind="overlay")
        io.imsave(segmented_image_filename, img_as_ubyte(colored_labels))

        log = f"""
Amyloid-β Plaque Analysis Research Log
======================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Image: {image_path}

Parameters:
- Threshold method: {threshold_method}
- Threshold value: {threshold_value if threshold_method != "adaptive" else "Adaptive local thresholding"}
- Minimum plaque size: {min_plaque_size} pixels²

Results:
- Total plaques detected: {len(plaque_data)}
- Avg plaque size: {df["Area"].mean():.2f} ± {df["Area"].std():.2f} pixels²
- Total plaque area: {total_plaque_area} pixels² ({area_fraction:.2f}% of tissue)
- Avg plaque intensity: {df["Mean_Intensity"].mean():.2f}

Outputs:
- Detailed CSV: {csv_filename}
- Segmented Image: {segmented_image_filename}
"""
    else:
        log = f"""
Amyloid-β Plaque Analysis Research Log
======================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Image: {image_path}

Parameters:
- Threshold method: {threshold_method}
- Threshold value: {threshold_value if threshold_method != "adaptive" else "Adaptive local thresholding"}
- Minimum plaque size: {min_plaque_size} pixels²

Results:
No plaques detected. Adjust threshold or minimum plaque size.
"""
    return log
