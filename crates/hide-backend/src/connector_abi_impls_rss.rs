//! `rss` connector: parse a committed RSS 2.0 fixture (no network).
//!
//! Read-only. Does **not** implement [`crate::connector_abi::connector::ConnectorWrite`].

use std::fs;
use std::path::PathBuf;

use crate::connector_abi::abi::{ConnectorAbi, FamilyId};
use crate::connector_abi::account::{AccountHandle, AccountStore, InFlightGuard};
use crate::connector_abi::connector::{
    BTreeMapStr, Connector, ConnectorObject, ConnectorRead, ListRequest, ReadRequest,
};
use crate::connector_abi::error::{ConnectorError, Result};
use crate::connector_abi::families;

/// One parsed feed item.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FeedItem {
    guid: String,
    title: String,
    link: String,
    description: String,
    pub_date: String,
}

/// Parsed feed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Feed {
    title: String,
    link: String,
    description: String,
    items: Vec<FeedItem>,
}

/// Minimal RSS 2.0 parser for committed fixtures. Not a general XML library —
/// deliberately small and network-free.
fn parse_rss_2(xml: &str) -> Result<Feed> {
    // Require channel.
    let channel = between(xml, "<channel>", "</channel>")
        .ok_or_else(|| ConnectorError::Parse("missing <channel> in RSS fixture".into()))?;
    let title = text_tag(channel, "title").unwrap_or_default();
    let link = text_tag(channel, "link").unwrap_or_default();
    let description = text_tag(channel, "description").unwrap_or_default();

    let mut items = Vec::new();
    let mut rest = channel;
    while let Some(item_body) = between(rest, "<item>", "</item>") {
        let item_title = text_tag(item_body, "title").unwrap_or_default();
        let item_link = text_tag(item_body, "link").unwrap_or_default();
        let item_desc = text_tag(item_body, "description").unwrap_or_default();
        let pub_date = text_tag(item_body, "pubDate").unwrap_or_default();
        let guid = text_tag(item_body, "guid")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                if !item_link.is_empty() {
                    item_link.clone()
                } else {
                    format!("item-{}", items.len())
                }
            });
        items.push(FeedItem {
            guid,
            title: item_title,
            link: item_link,
            description: item_desc,
            pub_date,
        });
        // Advance past this item.
        if let Some(idx) = rest.find("</item>") {
            rest = &rest[idx + "</item>".len()..];
        } else {
            break;
        }
    }
    Ok(Feed {
        title,
        link,
        description,
        items,
    })
}

fn between<'a>(s: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = s.find(open)? + open.len();
    let end = s[start..].find(close)? + start;
    Some(&s[start..end])
}

/// First direct-ish text tag content. Handles optional CDATA.
fn text_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let mut search = s;
    while let Some(idx) = search.find(&open) {
        let after = &search[idx + open.len()..];
        // Skip attributes to '>'
        let gt = after.find('>')?;
        let body_start = gt + 1;
        // Self-closing?
        if after[..gt].ends_with('/') {
            search = &after[body_start..];
            continue;
        }
        let body = &after[body_start..];
        let end = body.find(&close)?;
        let raw = body[..end].trim();
        let text = if let Some(inner) = between(raw, "<![CDATA[", "]]>") {
            inner.trim().to_string()
        } else {
            decode_entities(raw)
        };
        return Some(text);
    }
    None
}

fn decode_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Live RSS connector. Account credential material is the absolute path to a
/// feed XML file (fixture).
#[derive(Debug)]
pub struct RssConnector {
    abi: ConnectorAbi,
}

impl RssConnector {
    pub fn new() -> Self {
        Self {
            abi: families::rss(),
        }
    }

    fn load_feed(&self, handle: &AccountHandle) -> Result<Feed> {
        let path = PathBuf::from(handle.credential_material());
        if path.as_os_str().is_empty() {
            return Err(ConnectorError::InvalidRequest(
                "rss account feed path is empty".into(),
            ));
        }
        let xml = fs::read_to_string(&path)
            .map_err(|e| ConnectorError::Io(format!("read feed {}: {e}", path.display())))?;
        parse_rss_2(&xml)
    }
}

impl Default for RssConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for RssConnector {
    fn family_id(&self) -> &FamilyId {
        &self.abi.family_id
    }
    fn abi(&self) -> &ConnectorAbi {
        &self.abi
    }
}

impl ConnectorRead for RssConnector {
    fn list(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ListRequest,
    ) -> Result<Vec<ConnectorObject>> {
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let feed = self.load_feed(handle)?;
        let mut out = Vec::new();
        // Feed itself as first object when no prefix filter.
        if request.prefix.is_none() {
            let mut meta = BTreeMapStr::new();
            meta.insert("link", feed.link.clone());
            out.push(ConnectorObject {
                id: "feed".into(),
                object_type: "feed".into(),
                title: feed.title.clone(),
                content: Some(feed.description.clone()),
                metadata: meta,
            });
        }
        for item in feed.items {
            if let Some(pref) = &request.prefix {
                if !item.guid.contains(pref.as_str()) && !item.title.contains(pref.as_str()) {
                    continue;
                }
            }
            let mut meta = BTreeMapStr::new();
            meta.insert("link", item.link);
            meta.insert("pub_date", item.pub_date);
            out.push(ConnectorObject {
                id: item.guid,
                object_type: "item".into(),
                title: item.title,
                content: Some(item.description),
                metadata: meta,
            });
            if out.len() >= request.limit {
                break;
            }
        }
        guard.complete(store)?;
        Ok(out)
    }

    fn fetch(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ReadRequest,
    ) -> Result<ConnectorObject> {
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let feed = self.load_feed(handle)?;
        if request.locator == "feed" || request.locator.is_empty() {
            let mut meta = BTreeMapStr::new();
            meta.insert("link", feed.link);
            let obj = ConnectorObject {
                id: "feed".into(),
                object_type: "feed".into(),
                title: feed.title,
                content: Some(feed.description),
                metadata: meta,
            };
            guard.complete(store)?;
            return Ok(obj);
        }
        for item in feed.items {
            if item.guid == request.locator || item.link == request.locator {
                let mut meta = BTreeMapStr::new();
                meta.insert("link", item.link);
                meta.insert("pub_date", item.pub_date);
                let obj = ConnectorObject {
                    id: item.guid,
                    object_type: "item".into(),
                    title: item.title,
                    content: Some(item.description),
                    metadata: meta,
                };
                guard.complete(store)?;
                return Ok(obj);
            }
        }
        Err(ConnectorError::NotFound(request.locator.clone()))
    }
}

// Deliberately no `impl ConnectorWrite for RssConnector`.

#[cfg(test)]
mod parse_tests {
    use super::*;
    #[test]
    fn parses_minimal_rss() {
        let xml = r#"<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>T</title><link>http://example.test/</link><description>D</description>
<item>
<title>Item One</title>
<link>http://example.test/1</link>
<guid>guid-1</guid>
<description>Hello</description>
<pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
</item>
</channel></rss>"#;
        let feed = parse_rss_2(xml).unwrap();
        assert_eq!(feed.title, "T");
        assert_eq!(feed.items.len(), 1);
        assert_eq!(feed.items[0].guid, "guid-1");
        assert_eq!(feed.items[0].title, "Item One");
    }
}
