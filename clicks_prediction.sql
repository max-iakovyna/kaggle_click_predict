DROP MATERIALIZED VIEW clicks_subsample;
DROP MATERIALIZED VIEW topics_view CASCADE;
DROP MATERIALIZED VIEW categories_view CASCADE;

CREATE MATERIALIZED VIEW clicks_subsample AS (
	SELECT * FROM clicks_train LIMIT 1000000  
);
CREATE INDEX ON clicks_subsample (display_id);
CREATE INDEX ON clicks_subsample (ad_id);

 
CREATE MATERIALIZED VIEW topics_view AS (
  WITH indsc AS (
    SELECT
      row_number() OVER () AS indx,
      topic_id
    FROM documents_topics
    GROUP BY topic_id
  )
	SELECT
    document_id,
    string_agg(indsc.indx::text, ',')
  FROM
    documents_topics dt
  JOIN indsc ON (indsc.topic_id = dt.topic_id)
  GROUP BY document_id
); 
CREATE INDEX ON topics_view (document_id);

CREATE MATERIALIZED VIEW topics_view AS (

  WITH indsc AS (
    SELECT
      row_number() OVER () AS indx,
      category_id
    FROM documents_categories
    GROUP BY category_id
  )
  SELECT
    dc.document_id,
    string_agg(indsc.indx::text, ',')
  FROM
    documents_categories dc
  JOIN indsc ON (indsc.category_id = dc.category_id)
  GROUP BY document_id
);
CREATE INDEX ON categories_view (document_id);

DROP MATERIALIZED VIEW countries_view;
CREATE MATERIALIZED VIEW countries_view AS (
	SELECT 
		row_number() OVER () as index,
		split_part(events.geo_location, '>', 1) as state
	FROM events
	GROUP BY 
		split_part(events.geo_location, '>', 1)
	ORDER BY index ASC 
)
;
CREATE UNIQUE INDEX ON countries_view (state);


DROP MATERIALIZED VIEW provinces_view;
CREATE MATERIALIZED VIEW provinces_view AS (
	SELECT 
		row_number() OVER () as index,
		split_part(events.geo_location, '>', 2) as province
	FROM events
	GROUP BY 
		split_part(events.geo_location, '>', 2)
	ORDER BY index ASC
)
;
CREATE UNIQUE INDEX ON provinces_view (province);

DROP MATERIALIZED VIEW randomised_display_id CASCADE;
CREATE MATERIALIZED VIEW randomised_display_id AS (
  WITH rnd_clicks AS (
      SELECT *
      FROM clicks_train
      ORDER BY random()
      LIMIT 10000000
  )
  SELECT display_id
  FROM rnd_clicks
  GROUP BY display_id
  LIMIT 1000000
);
CREATE UNIQUE INDEX ON randomised_display_id (display_id);

	
DROP MATERIALIZED VIEW train_set;
CREATE MATERIALIZED VIEW train_set AS ( 
  SELECT 
  	
  	ck.display_id,
	ck.clicked as clicked,
	
	events.platform as platform,
				
	EXTRACT(DOW FROM to_timestamp(events.timestamp + 1465876799998) ) as dow,
	EXTRACT(MONTH FROM to_timestamp(events.timestamp + 1465876799998) ) as month,
	
	events.timestamp as ts,
	
	countries_view.index as state,
	provinces_view.index as province,
	split_part(events.geo_location, '>', 3) as addr,
	
	promoted_content.advertiser_id as advertiser,
	
	topics_view.string_agg as topics,
	categories_view.string_agg as categories
		
	
	FROM clicks_subsample ck

	JOIN events ON (ck.display_id = events.display_id)
	JOIN promoted_content ON (promoted_content.ad_id = ck.ad_id)
	JOIN topics_view ON (events.document_id = topics_view.document_id)
	JOIN categories_view ON (events.document_id = categories_view.document_id)
	JOIN countries_view ON (split_part(events.geo_location, '>', 1) = countries_view.state)
	JOIN provinces_view ON (split_part(events.geo_location, '>', 2) = provinces_view.province)
)
;


COPY (
SELECT

  ck.display_id,
  ck.clicked                                                         AS clicked,

  events.platform                                                    AS platform,

  EXTRACT(DOW FROM to_timestamp(events.timestamp + 1465876799998))   AS dow,
  EXTRACT(MONTH FROM to_timestamp(events.timestamp + 1465876799998)) AS month,

  events.timestamp                                                   AS ts,

  countries_view.index                                               AS state,
  provinces_view.index                                               AS province,
  split_part(events.geo_location, '>', 3)                            AS addr,

  promoted_content.advertiser_id                                     AS advertiser,

  topics_view.string_agg                                             AS topics,
  categories_view.string_agg                                         AS categories,

  dmp.source_id,
  dmp.publisher_id,

  dmc.source_id                                                      AS ad_source_id,
  dmc.publisher_id                                                   AS ad_publisher_id,

  twc.string_agg                                                     AS ad_topics,
  cwc.string_agg                                                     AS ad_categories

FROM clicks_subsample ck

  LEFT JOIN randomised_display_id ON (ck.display_id = randomised_display_id.display_id)
  LEFT JOIN events ON (ck.display_id = events.display_id)
  LEFT JOIN promoted_content ON (promoted_content.ad_id = ck.ad_id)
  LEFT JOIN topics_view ON (events.document_id = topics_view.document_id)
  LEFT JOIN categories_view ON (events.document_id = categories_view.document_id)
  LEFT JOIN countries_view ON (split_part(events.geo_location, '>', 1) = countries_view.state)
  LEFT JOIN provinces_view ON (split_part(events.geo_location, '>', 2) = provinces_view.province)
  LEFT JOIN documents_meta dmp ON (dmp.document_id = events.document_id)

  LEFT JOIN documents_meta dmc ON (dmc.document_id = promoted_content.document_id)
  LEFT JOIN topics_view twc ON (promoted_content.document_id = twc.document_id)
  LEFT JOIN categories_view cwc ON (promoted_content.document_id = cwc.document_id)

) TO 'c:\kaggel\kaggle_click_predict\data\train_set_3.csv' DELIMITER ',' CSV HEADER
;


SELECT count(*) FROM randomised_display_id;