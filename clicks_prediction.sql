SELECT 
	clicks_train.clicked as clicked,
	
	events.platform as platform,
	page_views_sample.traffic_source as traffic_source,
			
	EXTRACT(DOW FROM to_timestamp(events.timestamp + 1465876799998) ) as dow,
	EXTRACT(MONTH FROM to_timestamp(events.timestamp + 1465876799998) ) as month,
	EXTRACT(MONTH FROM to_timestamp(events.timestamp + 1465876799998) ) as month,
	
	events.timestamp as ts,
	
	split_part(events.geo_location, '>', 1) as state,
	split_part(events.geo_location, '>', 2) as province,
	split_part(events.geo_location, '>', 3) as addr,
	
	promoted_content.advertiser_id
	
	
FROM clicks_train
JOIN events ON (clicks_train.display_id = events.display_id)
JOIN page_views_sample ON (page_views_sample.uuid = events.uuid)
JOIN promoted_content ON (promoted_content.ad_id = clicks_train.ad_id)


  