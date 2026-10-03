// Example Cypher queries for the call knowledge graph (Neo4j Browser, http://localhost:7474).

// 1. Schema overview
CALL db.schema.visualization();

// 2. One call and its ontology entities
MATCH (c:Call {call_id: 'bothbosu_00012'})-[r]->(e) WHERE NOT e:Call
RETURN c, r, e;

// 3. The "government impersonation + gift card" script: calls that share both entities
MATCH (c:Call)-[:CLAIMS_TO_BE]->(:ClaimedIdentity {name: 'government_agent'}),
      (c)-[:PAY_VIA]->(:PaymentChannel {name: 'gift_card'})
RETURN c.call_id, c.split, c.gold_label
ORDER BY c.call_id LIMIT 50;

// 4. Scam rate per requested asset (training calls only)
MATCH (c:Call {split: 'train'})-[:REQUESTS]->(a:RequestedAsset)
RETURN a.name AS asset, count(c) AS calls, avg(toFloat(c.gold_label)) AS scam_rate
ORDER BY scam_rate DESC;

// 5. Identity -> tactic -> asset -> payment paths shared by the most calls
MATCH (c:Call)-[:CLAIMS_TO_BE]->(i), (c)-[:USES_TACTIC]->(t), (c)-[:REQUESTS]->(a), (c)-[:PAY_VIA]->(p)
RETURN i.name AS identity, t.name AS tactic, a.name AS asset, p.name AS payment, count(DISTINCT c) AS calls
ORDER BY calls DESC LIMIT 20;

// 6. Louvain communities of the call-call similarity graph (after build-graph --backend neo4j)
MATCH (c:Call) WHERE c.community IS NOT NULL
RETURN c.community AS community, count(*) AS size,
       avg(CASE WHEN c.split = 'train' THEN toFloat(c.gold_label) END) AS train_scam_rate
ORDER BY size DESC LIMIT 20;

// 7. Nearest neighbours of a call in the SIMILAR_TO graph
MATCH (c:Call {call_id: 'bothbosu_00012'})-[s:SIMILAR_TO]-(o:Call)
RETURN o.call_id, o.split, o.gold_label, s.score ORDER BY s.score DESC LIMIT 10;

// 8. Call-back numbers shared by several calls (possible scam operations)
MATCH (n:CallbackNumber)<-[:PROVIDES_NUMBER]-(c:Call)
WITH n, collect(c.call_id) AS calls WHERE size(calls) > 1
RETURN n.name, size(calls) AS n_calls, calls ORDER BY n_calls DESC;
