SOURCE_FILE=../Datasets/texas-latest.osm.pbf
SA_FILE=sa_downtown.pbf
SA_ROAD_FILE=roads.pbf
SA_BUILDING_FILE=buildings.pbf

[ ! -f "$SA_FILE" ] && osmium extract -b -98.5149,29.4441,-98.4734,29.3876 $SOURCE_FILE -o $SA_FILE

[ ! -f "$SA_ROAD_FILE" ] && osmium tags-filter $SA_FILE w/highway=unclassified,residential,living_street,service,motorway,trunk,primary,secondary,tertiary,unclassified_link,residential_link,living_street_link,service_link,motorway_link,trunk_link,primary_link,secondary_link,tertiary_link  -o $SA_ROAD_FILE

[ ! -f "$SA_BUILDING_FILE" ] && osmium tags-filter $SA_FILE w/building r/building=multipolygon -o $SA_BUILDING_FILE
