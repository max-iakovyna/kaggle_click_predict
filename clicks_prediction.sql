DROP MATERIALIZED VIEW clicks_subsample;
DROP MATERIALIZED VIEW topics_view;
DROP MATERIALIZED VIEW categories_view;
DROP MATERIALIZED VIEW train_set;
DROP MATERIALIZED VIEW countries_view;

CREATE MATERIALIZED VIEW clicks_subsample AS (
	SELECT * FROM clicks_train LIMIT 1000000  
);
CREATE INDEX ON clicks_subsample (display_id);
CREATE INDEX ON clicks_subsample (ad_id);

 
CREATE MATERIALIZED VIEW topics_view AS (
	SELECT document_id, string_agg(topic_id::text, ',') FROM documents_topics GROUP BY document_id
); 
CREATE INDEX ON clicks_subsample (document_id);

CREATE MATERIALIZED VIEW categories_view AS (
	SELECT document_id, string_agg(category_id::text, ',') FROM documents_categories GROUP BY document_id
);
CREATE INDEX ON categories_view (document_id);

CREATE MATERIALIZED VIEW countries_view AS (
	SELECT 
		split_part(events.geo_location, '>', 1) as state
	FROM events
	GROUP BY 
		split_part(events.geo_location, '>', 1)
	ORDER BY state ASC 
)
;
CREATE UNIQUE INDEX ON countries_view (state);


DROP MATERIALIZED VIEW provinces_view;
CREATE MATERIALIZED VIEW provinces_view AS (
	SELECT 
		split_part(events.geo_location, '>', 2) as province
	FROM events
	GROUP BY 
		split_part(events.geo_location, '>', 2)
	ORDER BY province ASC
)
;
CREATE UNIQUE INDEX ON provinces_view (province);
	

CREATE MATERIALIZED VIEW train_set AS ( 
  SELECT 
  	
	ck.clicked as clicked,
	
	events.platform as platform,
				
	EXTRACT(DOW FROM to_timestamp(events.timestamp + 1465876799998) ) as dow,
	EXTRACT(MONTH FROM to_timestamp(events.timestamp + 1465876799998) ) as month,
	
	events.timestamp as ts,
	
	split_part(events.geo_location, '>', 1) as state,
	split_part(events.geo_location, '>', 2) as province,
	split_part(events.geo_location, '>', 3) as addr,
	
	promoted_content.advertiser_id as advertiser,
	
	topics_view.string_agg as topics,
	categories_view.string_agg as categories
		
	
	FROM clicks_subsample ck
	
	JOIN events ON (ck.display_id = events.display_id)	
	JOIN promoted_content ON (promoted_content.ad_id = ck.ad_id)
	JOIN topics_view ON (events.document_id = topics_view.document_id)
	JOIN categories_view ON (events.document_id = categories_view.document_id)
)
;

