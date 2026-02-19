ARTFAM_QUERY = """
WITH historic_split_orders AS (
    SELECT vv."ERDAT", vv."BSTNK", COUNT(DISTINCT vv."VBELN") AS n_splits
    FROM sap.vbak_vbap vv
    WHERE (vv."ABGRU" != 'ZF' OR vv."ABGRU" IS NULL)
    GROUP BY vv."ERDAT", vv."BSTNK"
    HAVING COUNT(DISTINCT vv."VBELN") > 1
)
SELECT
    NULLIF(regexp_replace(zkqa."KLASSE", '_.*$', ''), '--') AS klasse,
    gor.description AS "description",
    zkqa."ARTFAM_ROHPLATTE" AS artfam_rohplatte,
    vv."MATNR" AS sku
FROM
    prediction.line_item_result lir
INNER JOIN
    prediction.order_result or2 ON or2.id = lir.order_result_id
INNER JOIN
    prediction.attachment a ON a.id = or2.attachment_id
INNER JOIN
    prediction.email e ON e.id = a.email_id
INNER JOIN
    sap.vbak_vbap vv ON vv."BSTNK" = or2.bstkd
        AND (vv."POSNR"::integer / 10) = lir.posnr
        AND TO_DATE(vv."ERDAT", 'DD.MM.YYYY') BETWEEN (or2.bstdk::date - INTERVAL '5 day') AND (or2.bstdk::date + INTERVAL '5 day')
INNER JOIN
    sap.zzzcu50_komk_q_aus zkqa ON zkqa."MATNR" = vv."MATNR"
INNER JOIN
    prediction.gpt_output_result gor ON gor.line_item_result_id = lir.id
WHERE
    or2.bstkd NOT IN (SELECT \"BSTNK\" FROM historic_split_orders)
    AND LENGTH(or2.bstkd::text) > 5
"""

CORE_QUERY = """
WITH historic_split_orders AS (
    SELECT vv."ERDAT", vv."BSTNK", COUNT(DISTINCT vv."VBELN") AS n_splits
    FROM sap.vbak_vbap vv
    WHERE (vv."ABGRU" != 'ZF' OR vv."ABGRU" IS NULL)
    GROUP BY vv."ERDAT", vv."BSTNK"
    HAVING COUNT(DISTINCT vv."VBELN") > 1
)
SELECT
    NULLIF(regexp_replace(zkqa."KLASSE", '_.*$', ''), '--') AS core,
    gor.description AS "description"
FROM
    prediction.line_item_result lir
INNER JOIN
    prediction.order_result or2 ON or2.id = lir.order_result_id
INNER JOIN
    prediction.attachment a ON a.id = or2.attachment_id
INNER JOIN
    prediction.email e ON e.id = a.email_id
INNER JOIN
    sap.vbak_vbap vv ON vv."BSTNK" = or2.bstkd
        AND (vv."POSNR"::integer / 10) = lir.posnr
        AND TO_DATE(vv."ERDAT", 'DD.MM.YYYY') BETWEEN (or2.bstdk::date - INTERVAL '5 day') AND (or2.bstdk::date + INTERVAL '5 day')
INNER JOIN
    sap.zzzcu50_komk_q_aus zkqa ON zkqa."MATNR" = vv."MATNR"
INNER JOIN
    prediction.gpt_output_result gor ON gor.line_item_result_id = lir.id
WHERE
    or2.bstkd NOT IN (SELECT \"BSTNK\" FROM historic_split_orders)
    AND LENGTH(or2.bstkd::text) > 5
"""

ROH_QUERY = """
WITH historic_split_orders AS (
	SELECT vv."ERDAT", vv."BSTNK", COUNT(DISTINCT vv."VBELN" ) AS n_splits
	FROM sap.vbak_vbap vv
	where (vv."ABGRU" != 'ZF' or vv."ABGRU" is null)
	GROUP BY vv."ERDAT", vv."BSTNK"
	HAVING COUNT(DISTINCT vv."VBELN" ) > 1
)

SELECT
	NULLIF(regexp_replace(zkqa."KLASSE", '_.*$', ''), '--') AS klasse,
	gor.description as description,
	zkqa."ARTFAM_ROHPLATTE" as artfam_rohplatte,
	vv."MATNR" as sku,
    zkqa."DEKOR_VS"  as dekor_vs, 
	zkqa."DEKOR_RS"  as dekor_rs
FROM
    prediction.line_item_result lir
INNER JOIN
    prediction.order_result or2 ON or2.id = lir.order_result_id
INNER JOIN
    prediction.attachment a ON a.id = or2.attachment_id
INNER JOIN
    prediction.email e ON e.id = a.email_id
INNER JOIN
    sap.vbak_vbap vv ON vv."BSTNK" = or2.bstkd
        AND (vv."POSNR"::integer / 10) = lir.posnr
        AND TO_DATE(vv."ERDAT", 'DD.MM.YYYY') BETWEEN (or2.bstdk::date - INTERVAL '5 day') AND (or2.bstdk::date + INTERVAL '5 day')
INNER JOIN
    sap.zzzcu50_komk_q_aus zkqa ON zkqa."MATNR" = vv."MATNR"
INNER JOIN
    prediction.gpt_output_result gor ON gor.line_item_result_id = lir.id
WHERE
    or2.bstkd NOT IN (SELECT \"BSTNK\" FROM historic_split_orders)
    AND LENGTH(or2.bstkd::text) > 5
"""