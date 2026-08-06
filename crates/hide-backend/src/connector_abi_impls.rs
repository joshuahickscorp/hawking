//! Real connector implementations (fixture-backed, no network).

pub use local_folder::LocalFolderConnector;
pub use rss::RssConnector;

#[path = "connector_abi_impls_local_folder.rs"]
pub mod local_folder;
#[path = "connector_abi_impls_rss.rs"]
pub mod rss;
