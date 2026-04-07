"""imagebreed unified analytics package."""

from .config import OrthoServerConfig
from .io import download_to_cache, inspect_geotiff, cutout_band_by_polygon, cutout_array_by_polygon, save_cutout
from .geometry import find_area_feature, find_child_feature, get_children_for_area, list_area_labels, collect_all_area_features
from .calibration import scale_to_reflectance, compute_pif_stats, calibrate_band_to_reference, calibrate_all_bands
from .indices import ndvi, gndvi, ndre, evi, savi, msavi, gli, make_vegetation_mask, make_vegetation_mask_msavi, apply_vegetation_mask, MsaviMaskResult, BUILTIN_INDICES
from .display import normalize_for_display, rgb_composite, false_color_composite, apply_mask_overlay, apply_mask_blackout, draw_contours
from .analysis import AreaResult, process_all_areas, analyze_field_from_server
from .export import results_to_csv, results_to_dataframe, append_results_to_csv
from .raster_mask import reproject_to_reference, png_blue_to_mask, save_mask_geotiff
from .plot import plot_mask_overlay, plot_masked
from .mask_transfer import (
    MaskTransferConfig,
    MaskTransferResult,
    MaskTransferPipelineConfig,
    MaskProjectionArtifacts,
    export_mask_as_geotiff,
    confirm_saved_mask,
    confirm_saved_mask_overlay,
    save_mask_overlay_png,
    save_overlay_rgb,
    validate_mask_grid,
    extract_area_cutout_and_mask,
    run_mask_transfer,
    build_mask_transfer_config,
    run_mask_transfer_pipeline,
    run_mask_projection_only,
)

# Consolidated legacy domains
from .api import OrthoCatalog, OrthoAPIClient, fetch_ortho_catalog, build_ortho_url, cutout_areas, load_geojson_polygons
from .vegetation import compute_vegetation_indices, reflectance_to_lab, apply_mask
from .grid import generate_grid, merge_grids, visualize_grid, export_csv, update_field_coords
from .legacy_compat import (
    _download_to_cache,
    cutout_band_by_polygon_URL_REMOTE,
    nd_index,
    make_veg_mask_ndvi,
    clean_mask,
    read_band_as_reflectance,
    get_one_child,
    get_child_by_name,
    find_child_feature_dict,
    find_area_feature_dict,
    get_children_feature_dicts_for_area,
    get_metadata_geo_transform_and_scale,
    ensure_projected_geometry_from_pixelpoints,
    print_ortho_catalog_summary,
    list_available_orthos,
    get_available_bands,
    get_locator_availability,
    print_locator_availability,
    scaled_pixel_points,
    plot_overlay_at_scale,
    rescale_pixel_points,
    scaled_pixel_points_xy,
    crop_area_cutouts,
    cutout_children,
    load_image_from_url,
    plot_geojson_pixelpoints_on_image,
    plot_geojson_pixelpoints_on_area_cutout_object,
    LocatorAvailability,
)

__all__ = [
    "OrthoServerConfig",
    "download_to_cache", "inspect_geotiff", "cutout_band_by_polygon", "cutout_array_by_polygon", "save_cutout",
    "find_area_feature", "find_child_feature", "get_children_for_area", "list_area_labels", "collect_all_area_features",
    "scale_to_reflectance", "compute_pif_stats", "calibrate_band_to_reference", "calibrate_all_bands",
    "ndvi", "gndvi", "ndre", "evi", "savi", "msavi", "gli", "make_vegetation_mask", "make_vegetation_mask_msavi", "apply_vegetation_mask", "MsaviMaskResult", "BUILTIN_INDICES",
    "normalize_for_display", "rgb_composite", "false_color_composite", "apply_mask_overlay", "apply_mask_blackout", "draw_contours",
    "AreaResult", "process_all_areas", "analyze_field_from_server", "results_to_csv", "results_to_dataframe", "append_results_to_csv",
    "reproject_to_reference", "png_blue_to_mask", "save_mask_geotiff", "plot_mask_overlay", "plot_masked",
    "MaskTransferConfig", "MaskTransferResult", "MaskTransferPipelineConfig", "MaskProjectionArtifacts", "export_mask_as_geotiff", "confirm_saved_mask", "confirm_saved_mask_overlay",
    "save_mask_overlay_png", "save_overlay_rgb", "validate_mask_grid", "extract_area_cutout_and_mask", "run_mask_transfer",
    "build_mask_transfer_config", "run_mask_transfer_pipeline", "run_mask_projection_only",
    "OrthoCatalog", "OrthoAPIClient", "fetch_ortho_catalog", "build_ortho_url", "cutout_areas", "load_geojson_polygons",
    "compute_vegetation_indices", "reflectance_to_lab", "apply_mask",
    "generate_grid", "merge_grids", "visualize_grid", "export_csv", "update_field_coords",
    "_download_to_cache", "cutout_band_by_polygon_URL_REMOTE", "nd_index", "make_veg_mask_ndvi", "clean_mask",
    "read_band_as_reflectance", "get_one_child", "get_child_by_name", "find_child_feature_dict", "find_area_feature_dict",
    "get_children_feature_dicts_for_area", "get_metadata_geo_transform_and_scale", "ensure_projected_geometry_from_pixelpoints",
    "print_ortho_catalog_summary", "list_available_orthos", "get_available_bands", "get_locator_availability",
    "print_locator_availability", "scaled_pixel_points", "plot_overlay_at_scale", "rescale_pixel_points", "scaled_pixel_points_xy",
    "crop_area_cutouts", "cutout_children", "load_image_from_url", "plot_geojson_pixelpoints_on_image",
    "plot_geojson_pixelpoints_on_area_cutout_object", "LocatorAvailability",
]
