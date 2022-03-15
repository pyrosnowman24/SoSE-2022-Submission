SOURCE_FILE=../Datasets/texas-latest.osm.pbf
ATX_FILE=austin_downtown.pbf
ATX_ROAD_FILE=austin_roads.pbf
ATX_BUILDING_FILE=austin_buildings.pbf

[ ! -f "$ATX_FILE" ] && osmium extract -b -97.7907,30.2330,-97.6664,30.3338 $SOURCE_FILE -o $ATX_FILE

[ ! -f "$ATX_ROAD_FILE" ] && osmium tags-filter $ATX_FILE w/highway=unclassified,residential,living_street,service,motorway,trunk,primary,secondary,tertiary,unclassified_link,residential_link,living_street_link,service_link,motorway_link,trunk_link,primary_link,secondary_link,tertiary_link  -o $ATX_ROAD_FILE

[ ! -f "$ATX_BUILDING_FILE" ] && osmium tags-filter $ATX_FILE w/building r/building=multipolygon -o $ATX_BUILDING_FILE
