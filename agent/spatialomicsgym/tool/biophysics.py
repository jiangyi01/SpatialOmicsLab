def predict_protein_disorder_regions(
    protein_sequence, threshold=0.5, output_file="disorder_prediction_results.csv", timeout=60
):
    """Predicts intrinsically disordered regions (IDRs) in a protein sequence using IUPred2A.

    Parameters
    ----------
    protein_sequence : str
        The amino acid sequence of the protein to analyze
    threshold : float, optional
        The disorder score threshold above which a residue is considered disordered (default: 0.5)
    output_file : str, optional
        Filename to save the per-residue disorder scores (default: "disorder_prediction_results.csv"; with
        the default, a later call writes disorder_prediction_results_2.csv, ... instead of overwriting)
    timeout : int, optional
        Seconds to wait for the IUPred2A server (default: 60)

    Returns
    -------
    str
        A research log summarizing the prediction process and results

    """
    import csv
    import re

    import requests

    # Clean the input sequence
    protein_sequence = "".join(re.findall(r"[A-Za-z]", protein_sequence))

    # Step 1: Submit the sequence to IUPred2A web server
    url = "https://iupred2a.elte.hu/iupred2a"
    payload = {
        "seq": protein_sequence,
        "iupred2": "long",  # Use IUPred2 long disorder prediction
        "anchor2": "no",  # Don't use ANCHOR2 prediction
    }

    # The POST had no timeout, so a stalled server held the request thread in recv indefinitely
    # (hunt 2026-09-30, uT2-pharmacology-20).
    try:
        response = requests.post(url, data=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        return f"Error: IUPred2A did not answer within timeout={timeout} s; raise timeout to wait longer ({e})"
    except requests.exceptions.RequestException as e:
        return f"Error accessing IUPred2A server: {str(e)}"

    # Step 2: Parse the results to extract disorder scores
    result_lines = response.text.split("\n")
    scores = []

    for line in result_lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                position = int(parts[0])
                residue = parts[1]
                score = float(parts[2])
                scores.append((position, residue, score))
            except (ValueError, IndexError):
                continue

    if not scores:
        return "No valid prediction data was returned from the server."

    # Step 3: Identify disordered regions
    disordered_regions = []
    current_region = []

    for pos, _, score in scores:
        if score >= threshold:
            if not current_region:
                current_region = [pos]
            elif pos == current_region[-1] + 1:
                current_region.append(pos)
            else:
                if len(current_region) > 1:
                    disordered_regions.append((current_region[0], current_region[-1]))
                current_region = [pos]
        elif current_region and len(current_region) > 1:
            disordered_regions.append((current_region[0], current_region[-1]))
            current_region = []
        elif current_region:
            current_region = []

    # Add the last region if it exists
    if current_region and len(current_region) > 1:
        disordered_regions.append((current_region[0], current_region[-1]))

    # Step 4: Save results to CSV. With the default name a second call overwrote the file the first log
    # names (hunt 2026-09-30, uT2-pharmacology-23).
    if output_file == "disorder_prediction_results.csv":
        from spatialomicsgym.tool.pharmacology import _fresh_output_path

        output_file = _fresh_output_path(output_file)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Position", "Amino_Acid", "Disorder_Score", "Is_Disordered"])
        for pos, aa, score in scores:
            is_disordered = "Yes" if score >= threshold else "No"
            writer.writerow([pos, aa, score, is_disordered])

    # Step 5: Generate the research log
    total_residues = len(scores)
    disordered_count = sum(1 for _, _, score in scores if score >= threshold)
    disordered_percentage = (disordered_count / total_residues) * 100 if total_residues > 0 else 0

    log = f"""
Intrinsically Disordered Region (IDR) Prediction Research Log:
=============================================================
Analysis performed using IUPred2A algorithm (long disorder mode)
Protein sequence length: {total_residues} amino acids
Disorder threshold: {threshold}

Results Summary:
- {disordered_count} residues ({disordered_percentage:.2f}%) predicted as disordered
- {len(disordered_regions)} distinct disordered regions identified

Disordered Regions:
"""

    if disordered_regions:
        for start, end in disordered_regions:
            length = end - start + 1
            log += f"- Region {start}-{end} (length: {length} residues)\n"
    else:
        log += "- No significant disordered regions found\n"

    log += f"\nDetailed per-residue scores saved to: {output_file}"

    return log


def analyze_cell_morphology_and_cytoskeleton(image_path, output_dir="./results", threshold_method="otsu"):
    """Quantifies cell morphology and cytoskeletal organization from fluorescence microscopy images.

    Parameters
    ----------
    image_path : str
        Path to the fluorescence microscopy image file
    output_dir : str, optional
        Directory to save output files (default: './results'; with the default, a later call's files get a
        _2, _3, ... suffix instead of overwriting an earlier result)
    threshold_method : str, optional
        Method for cell segmentation ('otsu', 'adaptive', or 'manual') (default: 'otsu')

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
    from skimage import exposure, feature, filters, io, measure, morphology
    from skimage.color import rgb2gray

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # With the default ./results a second call overwrote the files the first log names (hunt 2026-09-30,
    # uT2-pharmacology-23); an output_dir the caller names is used as given.
    morphology_names = ["cell_morphology_data.csv", "fiber_orientation_data.csv", "cell_segmentation.png"]
    if output_dir == "./results":
        from spatialomicsgym.tool.pharmacology import _fresh_output_paths

        cell_csv_path, fiber_csv_path, segmentation_path = _fresh_output_paths(output_dir, morphology_names)
    else:
        cell_csv_path, fiber_csv_path, segmentation_path = [os.path.join(output_dir, n) for n in morphology_names]

    # Start log
    log = f"Cell Morphology and Cytoskeleton Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Analyzing image: {image_path}\n\n"

    # Load image
    log += "Step 1: Loading and preprocessing image\n"
    try:
        image = io.imread(image_path)
        # Convert to grayscale if RGB
        if len(image.shape) > 2:
            gray_image = rgb2gray(image)
            log += "- Converted RGB image to grayscale\n"
        else:
            gray_image = image

        # Enhance contrast
        gray_image = exposure.equalize_hist(gray_image)
        log += "- Enhanced image contrast\n"
    except Exception as e:
        return f"Error loading image: {str(e)}"

    # Segment cells
    log += "\nStep 2: Segmenting cells from background\n"
    if threshold_method == "otsu":
        thresh = filters.threshold_otsu(gray_image)
        binary = gray_image > thresh
        log += f"- Applied Otsu thresholding (threshold value: {thresh:.4f})\n"
    elif threshold_method == "adaptive":
        binary = filters.threshold_local(gray_image, block_size=35, offset=0.05)
        binary = gray_image > binary
        log += "- Applied adaptive thresholding\n"
    else:  # manual
        thresh = 0.5  # Default value, can be parameterized
        binary = gray_image > thresh
        log += f"- Applied manual thresholding (threshold value: {thresh:.4f})\n"

    # Clean up binary image
    binary = morphology.remove_small_objects(binary, min_size=100)
    binary = morphology.remove_small_holes(binary, area_threshold=100)
    binary = morphology.binary_closing(binary, morphology.disk(3))
    log += "- Applied morphological operations to clean segmentation\n"

    # Label cells
    labeled_cells, num_cells = measure.label(binary, return_num=True)
    log += f"- Identified {num_cells} cell regions\n"

    # Analyze cell properties
    log += "\nStep 3: Analyzing cell morphology\n"
    cell_props = measure.regionprops_table(
        labeled_cells,
        gray_image,
        properties=(
            "area",
            "perimeter",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "orientation",
            "solidity",
        ),
    )

    # Convert to DataFrame for easier manipulation
    cell_df = pd.DataFrame(cell_props)

    # Calculate additional metrics
    if len(cell_df) > 0:
        cell_df["aspect_ratio"] = cell_df["major_axis_length"] / cell_df["minor_axis_length"]
        cell_df["circularity"] = (4 * np.pi * cell_df["area"]) / (cell_df["perimeter"] ** 2)

        # Summary statistics
        log += f"- Average cell area: {cell_df['area'].mean():.2f} pixels\n"
        log += f"- Average aspect ratio: {cell_df['aspect_ratio'].mean():.2f}\n"
        log += f"- Average circularity: {cell_df['circularity'].mean():.2f}\n"
        log += f"- Average eccentricity: {cell_df['eccentricity'].mean():.2f}\n"
    else:
        log += "- No cells detected for morphological analysis\n"

    # Analyze cytoskeletal organization
    log += "\nStep 4: Analyzing cytoskeletal organization\n"

    # Edge detection to highlight cytoskeletal fibers
    edges = feature.canny(gray_image, sigma=2)

    # Use Hough transform to detect lines (cytoskeletal fibers)
    if np.any(edges):
        lines = cv2.HoughLinesP(
            edges.astype(np.uint8),
            1,
            np.pi / 180,
            threshold=10,
            minLineLength=10,
            maxLineGap=5,
        )

        if lines is not None:
            # Calculate line orientations. OpenCV 5 returns segments as (N, 4), OpenCV 4 as (N, 1, 4);
            # line[0] was a scalar under 5 and the loop crashed. The `x2 - x1 != 0` guard dropped every
            # vertical fibre although arctan2 needs none (hunt 2026-09-30, uT2-pharmacology-29).
            orientations = []
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                angle = np.arctan2(float(y2 - y1), float(x2 - x1)) * 180 / np.pi
                orientations.append(angle)

            if orientations:
                # Convert to numpy array for calculations
                orientations = np.array(orientations)

                # Fibre segments are undirected (axial): 10 and -170 degrees are one axis, and their
                # arithmetic mean (-80) is perpendicular to it. Angles are averaged as doubled angles.
                # Normalize angles to -90 to 90 degrees
                norm_angles = np.mod(orientations + 90, 180) - 90
                rad_angles = np.radians(norm_angles)
                mean_cos2, mean_sin2 = np.mean(np.cos(2 * rad_angles)), np.mean(np.sin(2 * rad_angles))
                mean_orientation = np.degrees(0.5 * np.arctan2(mean_sin2, mean_cos2))

                # Order parameter (measure of alignment, 1 = perfectly aligned, 0 = random)
                order_parameter = np.sqrt(mean_cos2**2 + mean_sin2**2)
                # Axial circular standard deviation
                std_orientation = np.degrees(0.5 * np.sqrt(-2 * np.log(max(order_parameter, 1e-12))))

                log += f"- Detected {len(orientations)} cytoskeletal fibers\n"
                log += f"- Mean fiber orientation: {mean_orientation:.2f} degrees (axial mean, -90 to 90)\n"
                log += f"- Standard deviation of orientation: {std_orientation:.2f} degrees (axial circular)\n"
                log += f"- Order parameter (alignment): {order_parameter:.4f} (0=random, 1=aligned)\n"

                # Add fiber data to dataframe
                fiber_df = pd.DataFrame({"fiber_orientation": orientations})
            else:
                log += "- No fiber orientations could be calculated\n"
                fiber_df = pd.DataFrame()
        else:
            log += "- No cytoskeletal fibers detected\n"
            fiber_df = pd.DataFrame()
    else:
        log += "- No edges detected for cytoskeletal analysis\n"
        fiber_df = pd.DataFrame()

    # Save results
    log += "\nStep 5: Saving results\n"

    # Save cell morphology data
    if len(cell_df) > 0:
        cell_df.to_csv(cell_csv_path, index=False)
        log += f"- Cell morphology data saved to: {cell_csv_path}\n"

    # Save fiber orientation data
    if len(fiber_df) > 0:
        fiber_df.to_csv(fiber_csv_path, index=False)
        log += f"- Fiber orientation data saved to: {fiber_csv_path}\n"

    # Save segmentation image
    if num_cells > 0:
        io.imsave(segmentation_path, labeled_cells.astype(np.uint8) * 50)
        log += f"- Cell segmentation image saved to: {segmentation_path}\n"

    # Summary
    log += "\nAnalysis Summary:\n"
    log += f"- Processed image: {image_path}\n"
    log += f"- Detected {num_cells} cells\n"
    if len(cell_df) > 0:
        log += f"- Cell size range: {cell_df['area'].min():.1f} to {cell_df['area'].max():.1f} pixels\n"
        log += f"- Cell shape: average aspect ratio = {cell_df['aspect_ratio'].mean():.2f}\n"
    if "order_parameter" in locals():
        log += f"- Cytoskeletal organization: alignment parameter = {order_parameter:.4f}\n"

    return log


def analyze_tissue_deformation_flow(image_sequence, output_dir="results", pixel_scale=1.0):
    """Quantify tissue deformation and flow dynamics from microscopy image sequence.

    Parameters
    ----------
    image_sequence : list or numpy.ndarray
        Sequence of microscopy images (either a list of file paths or a 3D numpy array [time, height, width])
    output_dir : str, optional
        Directory to save results (default: "results")
    pixel_scale : float, optional
        Physical scale of pixels (e.g., μm/pixel) for the reported displacement (default: 1.0);
        divergence, curl and strain are dimensionless and do not depend on it

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import cv2
    import numpy as np

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load images if paths are provided
    if isinstance(image_sequence[0], str):
        loaded_images = []
        for img_path in image_sequence:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return f"Error: Could not load image {img_path}"
            loaded_images.append(img)
        frames = np.array(loaded_images)
    else:
        # np.asarray: a list of 2-D arrays has no .shape and crashed here (hunt 2026-09-30, uT2-pharmacology-16)
        frames = np.asarray(image_sequence)
        # Convert to grayscale if needed
        if frames.ndim > 3:  # Has color channels
            frames = np.array(
                [
                    cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if frame.shape[-1] == 3 else cv2.COLOR_RGBA2GRAY)
                    if frame.shape[-1] in (3, 4)
                    else frame[..., 0]
                    for frame in frames
                ]
            )
    if frames.ndim != 3 or len(frames) < 2:
        return (
            f"Error: tissue deformation needs at least 2 frames of 2-D images (got array shape {frames.shape}); "
            "pass a [time, height, width] stack or a list of image paths."
        )
    if frames.dtype != np.uint8:  # Lucas-Kanade tracks 8-bit images; one scale for the whole sequence
        lo, hi = float(np.nanmin(frames)), float(np.nanmax(frames))
        frames = np.clip((frames - lo) / ((hi - lo) or 1.0) * 255.0, 0, 255).astype(np.uint8)

    # Parameters for optical flow
    lk_params = {
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }

    # Metrics storage
    num_frames = len(frames)
    flow_fields = []
    divergence_maps = []
    curl_maps = []
    strain_maps = []
    displacement_means = []

    # Log initialization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = f"Tissue Deformation and Flow Analysis - {timestamp}\n"
    log += f"Number of frames analyzed: {num_frames}\n"
    log += f"Image dimensions: {frames[0].shape[0]}x{frames[0].shape[1]} pixels\n"
    log += f"Pixel scale: {pixel_scale} units/pixel\n\n"
    log += "Analysis Steps:\n"

    # Create feature points grid (evenly spaced points)
    step = 20
    y, x = np.mgrid[0 : frames[0].shape[0] : step, 0 : frames[0].shape[1] : step]
    if min(y.shape) < 2:
        return f"Error: images of {frames[0].shape[0]}x{frames[0].shape[1]} px are too small for the {step}-px tracking grid."
    feature_points = np.stack((x.flatten(), y.flatten()), axis=1).astype(np.float32)

    for i in range(num_frames - 1):
        log += f"Processing frame pair {i} and {i + 1}...\n"

        # Calculate optical flow using Lucas-Kanade
        prev_frame = frames[i]
        next_frame = frames[i + 1]

        # Compute flow for the grid points
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, feature_points, None, **lk_params)

        # The displacement field lives on the tracking grid; a point that was not tracked is NaN.
        # It used to be written into an otherwise-zero full-resolution image, one pixel per grid point,
        # and differentiated with Sobel: the +/- spikes around each isolated point averaged to about zero
        # whatever the tissue did. Derivatives are now taken on the grid, over its spacing, in pixels per
        # pixel -- so divergence, curl and strain are dimensionless and pixel_scale (which used to divide
        # them) only scales the reported displacement (hunt 2026-09-30, uT2-pharmacology-16).
        valid = status.flatten() == 1
        displacement = np.full(feature_points.shape, np.nan)
        displacement[valid] = (next_points - feature_points)[valid]
        u = displacement[:, 0].reshape(y.shape)  # x-component of flow (pixels)
        v = displacement[:, 1].reshape(y.shape)  # y-component of flow (pixels)

        # Calculate derivatives for deformation analysis (central differences on the grid)
        u_y, u_x = np.gradient(u, step)
        v_y, v_x = np.gradient(v, step)

        # Calculate divergence (expansion/contraction)
        divergence = u_x + v_y

        # Calculate curl (rotation)
        curl = v_x - u_y

        # Calculate strain tensor components
        strain_xx = u_x
        strain_yy = v_y
        strain_xy = 0.5 * (u_y + v_x)

        # Magnitude of strain tensor (Frobenius norm)
        strain_magnitude = np.sqrt(strain_xx**2 + strain_yy**2 + 2 * strain_xy**2)

        # Store results
        flow_fields.append(np.dstack([u, v]))
        divergence_maps.append(divergence)
        curl_maps.append(curl)
        strain_maps.append(strain_magnitude)
        displacement_means.append(np.nanmean(np.hypot(u, v)) if valid.any() else np.nan)
        log += f"  - {int(valid.sum())} of {len(valid)} grid points tracked\n"

        # Visualize flow field
        flow_viz = np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8)
        flow_viz[..., 0] = next_frame  # Use next frame as background
        flow_viz[..., 1] = next_frame
        flow_viz[..., 2] = next_frame

        # Draw flow vectors for visualization
        for (px, py), (dx, dy) in zip(feature_points.astype(int), displacement, strict=True):
            if np.isfinite(dx) and (abs(dx) > 0.5 or abs(dy) > 0.5):  # Only draw significant flow
                cv2.arrowedLine(
                    flow_viz,
                    (int(px), int(py)),
                    (int(px + dx), int(py + dy)),
                    (0, 255, 0),  # Green
                    1,
                    tipLength=0.3,
                )

        # Save visualizations
        flow_viz_path = os.path.join(output_dir, f"flow_viz_{i:03d}.png")
        divergence_path = os.path.join(output_dir, f"divergence_{i:03d}.npy")
        curl_path = os.path.join(output_dir, f"curl_{i:03d}.npy")
        strain_path = os.path.join(output_dir, f"strain_{i:03d}.npy")

        cv2.imwrite(flow_viz_path, flow_viz)
        np.save(divergence_path, divergence)
        np.save(curl_path, curl)
        np.save(strain_path, strain_magnitude)

    if all(np.isnan(d).all() for d in divergence_maps):
        return (
            log
            + "\nError: no grid point could be tracked between any pair of frames, so no deformation was measured.\n"
        )

    # Calculate summary statistics (over the tracked grid points)
    mean_divergence = float(np.nanmean([np.nanmean(div) for div in divergence_maps]))
    max_divergence = float(np.nanmax([np.nanmax(div) for div in divergence_maps if not np.isnan(div).all()]))
    mean_curl = float(np.nanmean([np.nanmean(np.abs(c)) for c in curl_maps]))
    mean_strain = float(np.nanmean([np.nanmean(s) for s in strain_maps]))
    mean_displacement = float(np.nanmean(displacement_means)) * pixel_scale

    # Add summary to log
    log += "\nAnalysis Results (per frame interval; divergence, curl and strain are dimensionless):\n"
    log += f"Mean displacement: {mean_displacement:.6f} units (pixel_scale x pixels)\n"
    log += f"Mean tissue divergence: {mean_divergence:.6f} (expansion/contraction rate)\n"
    log += f"Maximum divergence: {max_divergence:.6f}\n"
    log += f"Mean absolute curl: {mean_curl:.6f} (rotation rate)\n"
    log += f"Mean strain magnitude: {mean_strain:.6f} (deformation intensity)\n\n"

    # Save summary data
    summary_data = {
        "mean_divergence": mean_divergence,
        "max_divergence": max_divergence,
        "mean_curl": mean_curl,
        "mean_strain": mean_strain,
        "mean_displacement": mean_displacement,
    }
    summary_path = os.path.join(output_dir, "deformation_summary.npy")
    np.save(summary_path, summary_data)

    log += "Files Generated:\n"
    log += f"- Flow visualization images: {output_dir}/flow_viz_*.png\n"
    log += f"- Divergence maps (on the {step}-px tracking grid): {output_dir}/divergence_*.npy\n"
    log += f"- Curl maps (on the {step}-px tracking grid): {output_dir}/curl_*.npy\n"
    log += f"- Strain maps (on the {step}-px tracking grid): {output_dir}/strain_*.npy\n"
    log += f"- Summary statistics: {output_dir}/deformation_summary.npy\n"

    return log
