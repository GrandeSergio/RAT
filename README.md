# Route Analysis Tool (RAT)

## Overview

The Route Analysis Tool (RAT) is a Python script that performs Linear Referencing System (LRS) operations on geospatial data such as nearest distance to object in specific mileage and generates a point Shapefile with located data along a specified route. This tool is designed to work with polygon, line, and point geometries and is particularly useful for tasks involving spatial analysis and route-based data extraction.

## Functionality

The RAT script provides the following functionality:

• Input Data Handling: The script accepts several input parameters, including paths to the input shapefile, additional distance layer shapefile, buffer layer shapefile, route shapefile, points measure shapefile, and more. These parameters define the input data sources and configuration for the analysis.

• Intersection and Conversion: The script performs geometric intersection operations between the input geometries and specified masks (buffer and route layers). Depending on the input geometry type (polygon, line, or point), it converts and intersects geometries as needed.

• Nearest Distance Calculation: RAT calculates the nearest distance from each point in the input data to the closest point in a specified point layer and adds the attributes of the closest points to the input GeoDataFrame.

• Data Extraction: The script extracts route data from the located features based on the specified root geometry type (polygon, line, or point). It calculates distances and organizes the data for further analysis or visualization.

• Data Cleanup (Optional): RAT offers the option to clean and organize the route data, particularly when dealing with polygon geometries. The cleaning process ensures data integrity and prepares it for reporting or mapping purposes.

## How to Use

To use the RAT script, follow these steps:

1. Clone the Repository: Clone or download the RAT script and its dependencies to your local machine.

2. Set Up Dependencies: Ensure that you have the required Python libraries installed, including GeoPandas, Shapely, and Pandas. You can install them using pip if they are not already installed:

   ```pip install geopandas shapely pandas```

3. Configure Input Parameters: Open the RAT script and configure the input parameters at the beginning of the script. These parameters include paths to shapefiles, route identifiers, and other options specific to your analysis.

4. Run the Script: Execute the RAT script using your preferred Python environment (e.g., Jupyter Notebook, command line). The script will perform the specified analysis and generate an output Shapefile containing the located data along the route.


## Example Usage

Here's an example of how to use the RAT script:

```
RAT(
    input_path="input_data.shp",
    dist_layer="distance_data.shp",
    buffer_layer="buffer_data.shp",
    route_path="route.shp",
    points_measure="points.shp",
    points_measure_column="measure_column",
    output_path="output_folder/",
    geometry_type="polygon",
    route_id="route_identifier",
    cleanup_data=True,
    high_precision_calculation=True,
    encoding="utf-8",
    crs="4326"
    )
```

## Dependencies

• GeoPandas

• Shapely

• Pandas

## License

This RAT script is provided under the MIT License. You are free to use, modify, and distribute it for your projects.

## Authors

Arkadiusz Dacewicz

## Feedback and Contributions
I welcome feedback and contributions to enhance the functionality and usability of the RAT script. Feel free to create issues, pull requests, or contact the authors for suggestions or improvements.

