use crate::kg::KnowledgeNode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiteratureMap {
    pub topic: String,
    pub papers: Vec<KnowledgeNode>,
    pub clusters: Vec<LiteratureCluster>,
    pub gaps: Vec<ResearchGap>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiteratureCluster {
    pub id: String,
    pub label: String,
    pub node_ids: Vec<String>,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchGap {
    pub id: String,
    pub description: String,
    pub supporting_node_ids: Vec<String>,
}
