//! Real connector implementations (fixture-backed, no network).

pub mod local_folder;
pub mod rss;

pub use local_folder::LocalFolderConnector;
pub use rss::RssConnector;
