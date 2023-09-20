import os
from geopandas import clip, GeoDataFrame
import pandas as pd
from shapely.geometry import MultiPoint
from shapely.ops import nearest_points
import warnings

os.environ['USE_PYGEOS'] = '0'
warnings.simplefilter(action='ignore', category=FutureWarning)


def RAT(input_path: str, dist_layer: str, buffer_layer: str, route_path: str, points_measure: str,
        points_measure_column: str, output_path: str, geometry_type: str, route_id: str,
        cleanup_data: bool = True, high_precision_calculation: bool = True, encoding: str = 'utf-8', crs: str = '4326'):
    """
    Perform Linear Referencing System (LRS) operations on geospatial data and return point Shapefile with located data.

    :param input_path: Path to the input shapefile.
    :param dist_layer: Path to the additional distance layer shapefile.
    :param buffer_layer: Path to the buffer layer shapefile.
    :param route_path: Path to the route shapefile.
    :param points_measure: Path to the points measure shapefile.
    :param points_measure_column: column in points_measure file which has measure values
    :param output_path: Path to save the output shapefile.
    :param geometry_type: Type of geometry in the input shapefile (only "polygon", "line", "point").
    :param route_id: Identifier for the route.
    :param cleanup_data: Whether you want to cleanup data (default is True).
    :param high_precision_calculation: Whether to use high precision for calculations (default is True).
    :param encoding: Encoding for reading shapefiles (default is 'utf-8').
    :param crs: Coordinate Reference System (CRS) for the output shapefile (default is '2180').
    """

    def intersect(in_file_gdf: GeoDataFrame, mask: GeoDataFrame) -> GeoDataFrame:
        """
        Perform a geometric intersection operation between in_file_gdf and a mask GeoDataFrame.

        :param in_file_gdf: Input GeoDataFrame to be intersected.
        :param mask: GeoDataFrame used as a mask for intersection.
        :return: GeoDataFrame containing the intersection results.
        """
        intersection_gdf = clip(in_file_gdf, mask)
        return intersection_gdf

    def calculate(in_file_gdf: GeoDataFrame, mask: GeoDataFrame, calculate_area: bool = False) -> GeoDataFrame:
        """
        Calculate the length or area of the intersection between the geometries in in_file_gdf and a mask GeoDataFrame.

        :param in_file_gdf: Input GeoDataFrame containing geometries to be calculated.
        :param mask: GeoDataFrame used as a mask for intersection and calculation.
        :param calculate_area: If True, calculate area; if False, calculate length.
        :return: GeoDataFrame with additional columns containing calculated length or area.
        """

        def calculate_length_area(geometry, calculate_area):
            if geometry.geom_type == 'LineString' or geometry.geom_type == 'MultiLineString':
                # Clip LineString using bounding box
                clipped_geometry = geometry.intersection(mask.unary_union)
                # Calculate length of the clipped LineString
                if not clipped_geometry.is_empty:
                    length = int(clipped_geometry.length) if not clipped_geometry.is_empty else 0
                    return length

            elif geometry.geom_type == 'Polygon' or geometry.geom_type == 'MultiPolygon':
                # Clip Polygon using intersection
                clipped_geometry = geometry.intersection(mask.unary_union)
                if not clipped_geometry.is_empty:
                    if calculate_area:
                        area = int(clipped_geometry.area) if not clipped_geometry.is_empty else 0
                        if area != 0:
                            return area
                    else:
                        length = int(clipped_geometry.length) if not clipped_geometry.is_empty else 0
                        return length

            # Return an empty dictionary if no calculation is performed
            return {}

        # Apply the calculate_length_area function to each row and add the results as new columns
        column_name = 'calc_area' if calculate_area else 'calc_len'
        in_file_gdf[column_name] = in_file_gdf['geometry'].apply(lambda x: calculate_length_area(x, calculate_area))

        return in_file_gdf

    def geometry_convert(input_gdf: GeoDataFrame,
                         line_gdf: GeoDataFrame,
                         dist_gdf: GeoDataFrame,
                         buffer_gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Perform geometry conversion and intersection operations on GeoDataFrames.

        :param input_gdf: Input GeoDataFrame (polygon, line, point) for processing.
        :param line_gdf: Line GeoDataFrame (LineString/MultiLineString) to create point intersections.
        :param dist_gdf: First mask (route scope) to define distances between mask1 and mask2.
        :param buffer_gdf: Second mask (search scope) to create intersections with input_layer.
        :return: GeoDataFrame with intersected and converted geometries.
        """

        # Check if line_gdf has LineString or MultiLineString geometry type
        if (line_gdf.geom_type == 'LineString').any() or (line_gdf.geom_type == 'MultiLineString').any():
            pass
        else:
            raise Exception(f'Invalid geometry type in parameter "line_gdf" in frame: {line_gdf.name}')

        input_gdf['root_id'] = input_gdf.index + 1
        input_gdf['scope_dist'] = input_gdf.geometry.apply(lambda g: min(dist_gdf.distance(g)))
        input_gdf['scope_dist'] = input_gdf['scope_dist'].apply(lambda x: '%2.0f' % x)
        input_gdf = calculate(in_file_gdf=input_gdf, mask=buffer_gdf)

        intersect_df = GeoDataFrame()

        # Calculate area or length of input_gdf
        if (input_gdf.geom_type == 'LineString').any() or (input_gdf.geom_type == 'MultiLineString').any():
            points_form_line = []
            for xindex, xlayer in input_gdf.iterrows():
                for yindex, ylayer in line_gdf.iterrows():
                    inter = xlayer['geometry'].intersection(ylayer['geometry'])
                    if inter:
                        xlayer_copy = xlayer.copy()
                        xlayer_copy['geometry'] = inter
                        points_form_line.append(xlayer_copy)
            inter = intersect(in_file_gdf=input_gdf, mask=buffer_gdf)
            points = inter.copy()
            points = points.explode(index_parts=True)
            points.geometry = points.geometry.apply(lambda x: MultiPoint(list(x.coords)))
            line_df = GeoDataFrame(points_form_line)
            intersect_df = intersect_df.append(points)
            intersect_df = intersect_df.append(line_df)

        elif (input_gdf.geom_type == 'Polygon').any() or (input_gdf.geom_type == 'MultiPolygon').any():
            inter = intersect(in_file_gdf=input_gdf, mask=buffer_gdf)
            inter = inter.explode(index_parts=True)
            inter['inter_id'] = range(1, len(inter) + 1)

            groups = inter.groupby(['root_id'])
            for group_key in groups.groups:
                group = groups.get_group(group_key)
                line_inter = intersect(in_file_gdf=group, mask=line_gdf)
                if not line_inter.empty:
                    line_inter = line_inter.explode(index_parts=True)
                    points_line = line_inter.copy()
                    points_line.geometry = points_line.geometry.apply(
                        lambda x: MultiPoint(list([x.coords[0], x.coords[-1]])))
                    intersect_df = pd.concat([intersect_df, points_line])
                else:
                    points = group.copy()
                    points.geometry = points.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
                    intersect_df = pd.concat([intersect_df, points])

        elif (input_gdf.geom_type == 'Point').any() or (input_gdf.geom_type == 'MultiPoint').any():
            inter = intersect(in_file_gdf=input_gdf, mask=buffer_gdf)
            inter = inter.explode(index_parts=True)
            intersect_df = intersect_df.append(inter)
        else:
            raise Exception(f'Invalid geometry type in parameter "line_layer" in file: {input_gdf}')

        intersect_df['Object_ID'] = [x + 1 for x in range(len(intersect_df))]
        out_df = GeoDataFrame(intersect_df).reset_index(drop=True)
        out_df = out_df.explode(column='geometry', index_parts=True)
        out_df['Object_ID'] = pd.factorize(out_df['root_id'])[0] + 1
        out_df['Point_ID'] = [x + 1 for x in range(len(out_df))]
        out_df['Distance'] = out_df.apply(lambda row: row['geometry'].distance(line_gdf.geometry).min(), axis=1)
        out_df['Distance'] = out_df['Distance'].apply(lambda x: '%.2f' % x)
        out_df['root_id'] = out_df['root_id'].map(lambda x: '%2.0f' % x)

        return out_df

    def calculate_nearest_distance(input_gdf: GeoDataFrame, point_gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Calculate the nearest distance from each point in input_gdf to the closest point in point_gdf,
        and add the attributes of the closest point to the input GeoDataFrame.

        :param input_gdf: Input GeoDataFrame containing points for which the nearest distances are calculated.
        :param point_gdf: GeoDataFrame containing points to which distances are calculated.
        :return: GeoDataFrame with additional attributes from the closest points in point_gdf.
        """

        # Create a copy of the input_gdf to avoid modifying the original data
        result_gdf = input_gdf.copy()

        # Iterate through each row in the input_gdf
        for idx, row in result_gdf.iterrows():
            input_point = row.geometry
            nearest_point = nearest_points(input_point, point_gdf.unary_union)[1]

            # Iterate through each column in the point_gdf and add their values to the result_gdf
            for column in point_gdf.columns:
                if column != 'geometry':
                    result_gdf.at[idx, column] = point_gdf.loc[point_gdf.distance(nearest_point).idxmin()][column]

        return result_gdf

    def extract_route_data(input_gdf: GeoDataFrame, root_geometry_type: str, measure_column: str,
                           high_precision: bool = True) -> GeoDataFrame:
        """
        Extract route data from an input GeoDataFrame based on the specified root geometry type.

        :param input_gdf: Input GeoDataFrame containing located features.
        :param root_geometry_type: Type of geometry from the root layer. Options: "polygon", "line", "point".
        :param high_precision: Whether to use high precision when extracting data (default is False).
        :return: Extracted GeoDataFrame.
        """

        # Reset the index to avoid the ValueError
        frame = input_gdf.reset_index(drop=True)

        # Check if the input GeoDataFrame is empty
        if frame.empty:
            return frame

        # Convert measure_column column to integers
        frame['Measure'] = frame[measure_column].astype(int)

        # Create a column with measure in kilometers
        frame['Mileage'] = (frame['Measure'] / 1000).round(3)

        # Create a column with absolute value of distance
        frame['Distance'] = frame['Distance'].astype(float)
        frame['Route_dist'] = frame['Distance'].abs()

        # Initialize the 'Side' column
        frame['Side'] = 'left/right'

        # Update 'Side' based on distance values
        frame.loc[(frame['Distance'] > 0.01), 'Side'] = 'left'
        frame.loc[(frame['Distance'] < -0.01), 'Side'] = 'right'

        # Convert 'Mileage' column to a numeric column 'km'
        frame['km'] = pd.to_numeric(frame['Mileage'])

        # Convert 'Mileage' column to a formatted string
        frame['Mileage'] = frame['Mileage'].apply(lambda x: '{:.0f}+{:03.0f}'.format(int(x), (x % 1) * 1000))

        dfx = []

        for group_key, group in frame.groupby(['Object_ID']):
            dist_min = group.loc[group['Side'] == 'left/right']
            dist_other = group.loc[group['Side'].isin(['left', 'right'])]

            if root_geometry_type == 'polygon':
                if not dist_min.empty:
                    route_groups = dist_min.groupby(route_id)
                    for route_key, route_group in route_groups:
                        if high_precision:
                            high_precision_group = route_group.groupby('inter_id')
                            for precision_group_key, precision_group in high_precision_group:
                                km_min = precision_group.loc[precision_group['km'].idxmin()]
                                df_km_min = pd.DataFrame([km_min])
                                km_max = precision_group.loc[precision_group['km'].idxmax()]
                                df_km_max = pd.DataFrame([km_max])
                                dfx.extend([df_km_min, df_km_max])
                        else:
                            km_min = route_group.loc[route_group['km'].idxmin()]
                            df_km_min = pd.DataFrame([km_min])
                            km_max = route_group.loc[route_group['km'].idxmax()]
                            df_km_max = pd.DataFrame([km_max])
                            dfx.extend([df_km_min, df_km_max])

                if all(group['Route_dist'] >= 0.5) and not dist_other.empty:
                    dist_minimum = dist_other.loc[dist_other['Route_dist'].idxmin()]
                    df_dist_minimum = pd.DataFrame([dist_minimum])
                    dfx.append(df_dist_minimum)

            elif root_geometry_type == 'line':
                if not dist_min.empty:
                    dfx.append(dist_min)

                if all(group['Route_dist'] >= 0.5) and not dist_other.empty:
                    dist_minimum = dist_other.loc[dist_other['Route_dist'].idxmin()]
                    df_dist_minimum = pd.DataFrame([dist_minimum])
                    dfx.append(df_dist_minimum)

            elif root_geometry_type == 'point':
                group_point = group.groupby(['Point_ID'])
                for point_key, point in group_point:
                    dist_point = point.loc[point['Side'].isin(['left', 'right'])]
                    dist_minimum = dist_point.loc[dist_point['Route_dist'].idxmin()]
                    df_dist_minimum = pd.DataFrame([dist_minimum])
                    dfx.append(df_dist_minimum)

            else:
                raise Exception(
                    f'Invalid argument {root_geometry_type} value. Select one of: "polygon", "line", "point"')

        out_frame = GeoDataFrame(pd.concat(dfx, ignore_index=True))
        out_frame['km'] = out_frame['km'].map(lambda x: '%2.3f' % x)
        out_frame['Route_dist'] = out_frame['Route_dist'].abs().astype(int)
        out_frame.drop_duplicates(subset=['Point_ID', 'geometry'], keep='first', inplace=True)
        return out_frame

    def cleanup_route_data(frame: GeoDataFrame, root_geom_type: str, high_precision: bool = False) -> GeoDataFrame:
        """
        Clean up and organize route data based on the specified root geometry type.

        :param frame: Input GeoDataFrame containing route data.
        :param root_geom_type: Type of root geometry. Options: "polygon", "line", "point".
        :param high_precision: Whether to use high precision when cleaning data (default is False).
        :return: Cleaned GeoDataFrame.
        """

        gdfx = GeoDataFrame()

        if root_geom_type == 'polygon':
            group1 = frame.groupby(['Object_ID'])
            for _, obj1 in group1:
                group2 = obj1.groupby([route_id])
                for _, obj2 in group2:
                    if len(obj2) >= 2:
                        if high_precision:
                            high_precision_group = obj2.groupby('inter_id')
                            for precision_group_key, precision_group in high_precision_group:
                                if len(precision_group) == 2:
                                    mpoints = precision_group.copy()
                                    mpoints['geometry'] = MultiPoint(precision_group['geometry'].explode().tolist())
                                    mpoints['km'] = pd.to_numeric(precision_group['km'])
                                    km_min = mpoints.loc[mpoints['km'].idxmin()]
                                    km_max = mpoints.loc[mpoints['km'].idxmax()]
                                    mpoints['Mileag'] = f"{km_min['Mileage']} - {km_max['Mileage']}"
                                    gdfx = pd.concat([gdfx, mpoints.head(1)])
                                else:
                                    mpoints = precision_group.copy()
                                    mpoints['geometry'] = mpoints['geometry'].apply(
                                        lambda x: MultiPoint(list(x.coords)))
                                    mpoints['Mileag'] = mpoints['Mileage']
                                    gdfx = pd.concat([gdfx, mpoints])
                        else:
                            mpoints = obj2.copy()
                            mpoints['geometry'] = MultiPoint(obj2['geometry'].explode().tolist())
                            obj2['km'] = pd.to_numeric(obj2['km'])
                            km_min = obj2.loc[obj2['km'].idxmin()]
                            km_max = obj2.loc[obj2['km'].idxmax()]
                            mpoints['Mileag'] = f"{km_min['Mileage']} - {km_max['Mileage']}"
                            gdfx = pd.concat([gdfx, mpoints.head(1)])
                    else:
                        points = obj2.copy()
                        points['geometry'] = points['geometry'].apply(lambda x: MultiPoint(list(x.coords)))
                        points['Mileag'] = points['Mileage']
                        gdfx = pd.concat([gdfx, points])
        else:
            frame['Mileag'] = frame['Mileage']
            gdfx = frame

        gdfx['km'] = pd.to_numeric(gdfx['km'])
        gdfx = gdfx.sort_values([route_id, 'km'], ascending=[True, True])
        gdfx = gdfx.drop(columns=['Point_ID', 'Measure', 'Distance', 'Mileage', 'km'])
        gdfx = gdfx.rename(columns={'Mileag': 'Mileage'})
        gdfx.insert(0, 'Lp', list(range(1, len(gdfx) + 1)))

        return gdfx

    def validate():
        # Validate files paths
        a_files = [input_path, route_path, buffer_layer, dist_layer]
        a_exist = [f for f in a_files if os.path.isfile(f)]
        a_non_exist = list(set(a_exist) ^ set(a_files))
        if len(a_non_exist) == 0:
            pass
        else:
            raise Exception('Wrong path to files: ' + ';\n'.join([str(lst) for lst in a_non_exist]))

        # create dataframes from shp and set crs to parameter projection
        input_gdf = GeoDataFrame.from_file(input_path, encoding=encoding).to_crs(crs)
        line_gdf = GeoDataFrame.from_file(route_path, encoding=encoding).to_crs(crs)
        dist_gdf = GeoDataFrame.from_file(dist_layer, encoding=encoding).to_crs(crs)
        buffer_gdf = GeoDataFrame.from_file(buffer_layer, encoding=encoding).to_crs(crs)
        points_gdf = GeoDataFrame.from_file(points_measure, encoding=encoding).to_crs(crs)
        input_gdf_base = os.path.basename(input_path)
        input_gdf.name = os.path.splitext(input_gdf_base)[0]
        line_gdf_base = os.path.basename(route_path)
        line_gdf.name = os.path.splitext(line_gdf_base)[0]

        print('Converting Data')
        convert = geometry_convert(input_gdf=input_gdf, line_gdf=line_gdf, dist_gdf=dist_gdf, buffer_gdf=buffer_gdf)
        print('Calculating Nearest Distance')
        calculations = calculate_nearest_distance(input_gdf=convert, point_gdf=points_gdf)
        print('Extracting Route Data')
        manipulate_data = extract_route_data(input_gdf=calculations, root_geometry_type=geometry_type,
                                             measure_column=points_measure_column,
                                             high_precision=high_precision_calculation)
        if not all(manipulate_data.is_empty):
            if cleanup_data == True:
                print('Cleaning data')
                func = cleanup_route_data(manipulate_data, geometry_type)
                print('Saving file')
                func.to_file(output_path + 'RAT_' + line_gdf.name[:3] + '_' + os.path.basename(input_path),
                             encoding=encoding)
            else:
                manipulate_data.to_file(
                    output_path + 'RAT_' + line_gdf.name[:3] + '_' + os.path.basename(input_path),
                    encoding=encoding)
        else:
            print(f'No intersection in Geodataframe: {input_gdf.name}')

    validate()
