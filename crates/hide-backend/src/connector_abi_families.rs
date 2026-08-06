//! Every YOU connector family ABI declaration.
//!
//! Only `local_folder` and `rss` are [`ImplementationStatus::Implemented`].
//! All other families are fully declared (ABI filled) but not constructible.

use crate::connector_abi::abi::{
    AuditPolicy, AuthMethod, ChangeTransport, ConnectorAbi, ConnectorScope, EffectClass, FamilyId,
    ImplementationStatus, ObjectType, OfflineCache, RateLimit, ReadCapability, RevocationPolicy,
    SyncMode, WriteCapability,
};

fn base(id: &str, name: &str, description: &str, status: ImplementationStatus) -> ConnectorAbi {
    ConnectorAbi {
        family_id: FamilyId::new(id),
        display_name: name.into(),
        description: description.into(),
        status,
        read: ReadCapability::list_and_fetch(),
        write: WriteCapability::none(),
        auth: AuthMethod::None,
        scopes: vec![],
        object_types: vec![],
        sync: SyncMode::FullOnly,
        change_transport: ChangeTransport::None,
        offline_cache: OfflineCache::none(),
        rate_limit: RateLimit::local(),
        effect_classes: vec![EffectClass::Read],
        revocation: RevocationPolicy::real_local(),
        audit: AuditPolicy::writes_required(),
        honesty_notes: String::new(),
    }
}

/// `local_folder` — real, fixture-backed directory reader.
pub fn local_folder() -> ConnectorAbi {
    let mut a = base(
        "local_folder",
        "Local Folder",
        "Read a local directory tree under an explicit root path bound to the account handle.",
        ImplementationStatus::Implemented,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::none(); // type boundary: no ConnectorWrite impl
    a.auth = AuthMethod::None;
    a.scopes = vec![ConnectorScope::read(
        "folder.read",
        "Read files and list directories under the account root",
    )];
    a.object_types = vec![
        ObjectType::new("file", "A regular file"),
        ObjectType::new("directory", "A directory entry"),
    ];
    a.sync = SyncMode::FullOnly;
    a.change_transport = ChangeTransport::LocalWatch;
    a.offline_cache = OfflineCache::none();
    a.rate_limit = RateLimit::local();
    a.effect_classes = vec![EffectClass::Read];
    a.revocation = RevocationPolicy::real_local();
    a.audit = AuditPolicy::writes_required();
    a.honesty_notes =
        "IMPLEMENTED against local filesystem. Read-only. No network. Account credential is the root path."
            .into();
    a
}

/// `rss` — real, fixture-backed feed parser.
pub fn rss() -> ConnectorAbi {
    let mut a = base(
        "rss",
        "RSS / Atom Feed",
        "Parse an RSS or Atom feed from a local fixture path or (declared) remote URL.",
        ImplementationStatus::Implemented,
    );
    a.read = ReadCapability::list_and_fetch();
    a.write = WriteCapability::none();
    a.auth = AuthMethod::None;
    a.scopes = vec![ConnectorScope::read("feed.read", "Read feed items")];
    a.object_types = vec![
        ObjectType::new("feed", "Feed metadata"),
        ObjectType::new("item", "A feed item / entry"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 300,
    };
    a.offline_cache = OfflineCache::full(8 * 1024 * 1024);
    a.rate_limit = RateLimit::local();
    a.effect_classes = vec![EffectClass::Read];
    a.revocation = RevocationPolicy::real_local();
    a.audit = AuditPolicy::writes_required();
    a.honesty_notes =
        "IMPLEMENTED against committed fixture XML only. No network fetch in this crate.".into();
    a
}

pub fn github() -> ConnectorAbi {
    let mut a = base(
        "github",
        "GitHub",
        "Repositories, issues, PRs, and file contents via the GitHub API.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: false,
    };
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://github.com/login/oauth/authorize".into(),
        token_url: "https://github.com/login/oauth/access_token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("repo", "Read repository contents and metadata"),
        ConnectorScope::write("repo", "Open PRs, comment, push (elevated)"),
        ConnectorScope::read("read:user", "Read user profile"),
    ];
    a.object_types = vec![
        ObjectType::new("repository", "A GitHub repository"),
        ObjectType::new("issue", "An issue"),
        ObjectType::new("pull_request", "A pull request"),
        ObjectType::new("file", "A file blob in a repo"),
        ObjectType::new("comment", "An issue or PR comment"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 60,
        verification: "github-hmac-sha256".into(),
    };
    a.offline_cache = OfflineCache::metadata(64 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(5000, 100, "GitHub REST primary rate limit (authenticated)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn google_drive() -> ConnectorAbi {
    let mut a = base(
        "google_drive",
        "Google Drive",
        "Files and folders in Google Drive.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url: "https://oauth2.googleapis.com/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("drive.readonly", "Read Drive files"),
        ConnectorScope::write("drive.file", "Create/update app-created files"),
    ];
    a.object_types = vec![
        ObjectType::new("file", "A Drive file"),
        ObjectType::new("folder", "A Drive folder"),
        ObjectType::new("revision", "A file revision"),
    ];
    a.sync = SyncMode::DeltaToken;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 60,
        verification: "google-push".into(),
    };
    a.offline_cache = OfflineCache::full(256 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(1000, 100, "Drive API per-user quota (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn icloud_drive() -> ConnectorAbi {
    let mut a = base(
        "icloud_drive",
        "iCloud Drive / Local File Provider",
        "iCloud Drive via local file-provider mount or CloudKit (declared).",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::LocalSecret;
    a.scopes = vec![
        ConnectorScope::read("icloud.read", "Read iCloud Drive paths"),
        ConnectorScope::write("icloud.write", "Write iCloud Drive paths"),
    ];
    a.object_types = vec![
        ObjectType::new("file", "A file"),
        ObjectType::new("directory", "A directory"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::LocalWatch;
    a.offline_cache = OfflineCache::full(512 * 1024 * 1024);
    a.rate_limit = RateLimit::local();
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::SecretAccess,
    ];
    a.revocation = RevocationPolicy::real_local();
    a.honesty_notes =
        "DECLARED only. Distinct from implemented local_folder; not constructible.".into();
    a
}

pub fn gmail() -> ConnectorAbi {
    let mut a = base(
        "gmail",
        "Gmail",
        "Gmail messages, threads, labels.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: true,
    };
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url: "https://oauth2.googleapis.com/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("gmail.readonly", "Read mail"),
        ConnectorScope::write("gmail.send", "Send mail"),
        ConnectorScope::write("gmail.modify", "Modify labels / trash"),
    ];
    a.object_types = vec![
        ObjectType::new("message", "An email message"),
        ObjectType::new("thread", "A message thread"),
        ObjectType::new("label", "A Gmail label"),
        ObjectType::new("attachment", "A message attachment"),
    ];
    a.sync = SyncMode::DeltaToken;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 30,
        verification: "google-push".into(),
    };
    a.offline_cache = OfflineCache::metadata(128 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(250, 25, "Gmail API quota (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn google_calendar() -> ConnectorAbi {
    let mut a = base(
        "google_calendar",
        "Google Calendar",
        "Calendars and events.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url: "https://oauth2.googleapis.com/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("calendar.readonly", "Read calendars and events"),
        ConnectorScope::write("calendar.events", "Create/update/delete events"),
    ];
    a.object_types = vec![
        ObjectType::new("calendar", "A calendar"),
        ObjectType::new("event", "A calendar event"),
    ];
    a.sync = SyncMode::DeltaToken;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 60,
    };
    a.offline_cache = OfflineCache::metadata(32 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(1000, 100, "Calendar API quota (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn google_contacts() -> ConnectorAbi {
    let mut a = base(
        "google_contacts",
        "Google Contacts",
        "People / contacts directory.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: true,
    };
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url: "https://oauth2.googleapis.com/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("contacts.readonly", "Read contacts"),
        ConnectorScope::write("contacts", "Mutate contacts"),
    ];
    a.object_types = vec![ObjectType::new("person", "A contact / person")];
    a.sync = SyncMode::DeltaToken;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 300,
    };
    a.offline_cache = OfflineCache::metadata(16 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(90, 10, "People API quota (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn slack() -> ConnectorAbi {
    let mut a = base(
        "slack",
        "Slack",
        "Channels, messages, files.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: true,
    };
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://slack.com/oauth/v2/authorize".into(),
        token_url: "https://slack.com/api/oauth.v2.access".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("channels:history", "Read channel history"),
        ConnectorScope::read("files:read", "Read files"),
        ConnectorScope::write("chat:write", "Post messages"),
    ];
    a.object_types = vec![
        ObjectType::new("channel", "A channel"),
        ObjectType::new("message", "A message"),
        ObjectType::new("file", "A shared file"),
        ObjectType::new("user", "A workspace user"),
    ];
    a.sync = SyncMode::Cursor;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 30,
        verification: "slack-signing-secret".into(),
    };
    a.offline_cache = OfflineCache::metadata(64 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(50, 10, "Slack tiered rate limits (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn notion() -> ConnectorAbi {
    let mut a = base(
        "notion",
        "Notion",
        "Pages, databases, blocks.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://api.notion.com/v1/oauth/authorize".into(),
        token_url: "https://api.notion.com/v1/oauth/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("notion.read", "Read pages and databases"),
        ConnectorScope::write("notion.write", "Create/update pages and blocks"),
    ];
    a.object_types = vec![
        ObjectType::new("page", "A Notion page"),
        ObjectType::new("database", "A database"),
        ObjectType::new("block", "A block"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 60,
    };
    a.offline_cache = OfflineCache::metadata(64 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(180, 3, "Notion ~3 rps average (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn dropbox_onedrive() -> ConnectorAbi {
    let mut a = base(
        "dropbox_onedrive",
        "Dropbox / OneDrive",
        "Cloud file storage via Dropbox or Microsoft Graph (OneDrive).",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://www.dropbox.com/oauth2/authorize".into(),
        token_url: "https://api.dropboxapi.com/oauth2/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("files.metadata.read", "List and metadata"),
        ConnectorScope::read("files.content.read", "Download content"),
        ConnectorScope::write("files.content.write", "Upload / modify"),
    ];
    a.object_types = vec![
        ObjectType::new("file", "A cloud file"),
        ObjectType::new("folder", "A cloud folder"),
    ];
    a.sync = SyncMode::Cursor;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 60,
        verification: "provider-webhook".into(),
    };
    a.offline_cache = OfflineCache::full(512 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(600, 50, "Provider-dependent (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. Dual-provider family; no live API, not constructible.".into();
    a
}

pub fn browser_search() -> ConnectorAbi {
    let mut a = base(
        "browser_search",
        "Browser and Search",
        "Web page fetch and search results (declared; no live crawl here).",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::none();
    a.auth = AuthMethod::None;
    a.scopes = vec![
        ConnectorScope::read("web.fetch", "Fetch a URL"),
        ConnectorScope::read("web.search", "Run a search query"),
    ];
    a.object_types = vec![
        ObjectType::new("page", "A web page capture"),
        ObjectType::new("search_result", "A search hit"),
    ];
    a.sync = SyncMode::FullOnly;
    a.change_transport = ChangeTransport::None;
    a.offline_cache = OfflineCache::full(128 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(60, 10, "Search provider limits (declared)");
    a.effect_classes = vec![EffectClass::Read, EffectClass::Network];
    a.revocation = RevocationPolicy::real_local();
    a.honesty_notes =
        "DECLARED only. No network. Distinct from hide-browser crate integration.".into();
    a
}

pub fn generic_mcp() -> ConnectorAbi {
    let mut a = base(
        "generic_mcp",
        "Generic MCP",
        "Model Context Protocol server as a connector (tools/resources).",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: true,
    };
    a.auth = AuthMethod::McpSession;
    a.scopes = vec![
        ConnectorScope::read("mcp.resources", "List and read MCP resources"),
        ConnectorScope::write("mcp.tools", "Invoke MCP tools (may mutate)"),
    ];
    a.object_types = vec![
        ObjectType::new("resource", "An MCP resource"),
        ObjectType::new("tool_result", "Result of an MCP tool call"),
        ObjectType::new("prompt", "An MCP prompt template"),
    ];
    a.sync = SyncMode::FullOnly;
    a.change_transport = ChangeTransport::None;
    a.offline_cache = OfflineCache::none();
    a.rate_limit = RateLimit::remote(120, 20, "Server-dependent (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::ExternalMutation,
        EffectClass::Network,
        EffectClass::SecretAccess,
    ];
    a.revocation = RevocationPolicy::real_local();
    a.honesty_notes = "DECLARED only. No MCP session runtime in this crate.".into();
    a
}

pub fn generic_oauth_api() -> ConnectorAbi {
    let mut a = base(
        "generic_oauth_api",
        "Generic OAuth / API",
        "Catch-all OAuth2 + REST connector template for user-configured APIs.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://example.invalid/oauth/authorize".into(),
        token_url: "https://example.invalid/oauth/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("api.read", "Read API resources"),
        ConnectorScope::write("api.write", "Mutate API resources"),
    ];
    a.object_types = vec![ObjectType::new("resource", "A generic API resource")];
    a.sync = SyncMode::Cursor;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 120,
    };
    a.offline_cache = OfflineCache::metadata(32 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(60, 10, "User-configured (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. Template family; not constructible.".into();
    a
}

pub fn hawking_artifact_registry() -> ConnectorAbi {
    let mut a = base(
        "hawking_artifact_registry",
        "Hawking Artifact Registry",
        "Local Hawking/HIDE artifact registry: receipts, models metadata, sealed packs.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: false,
        delete: false,
    };
    a.auth = AuthMethod::LocalSecret;
    a.scopes = vec![
        ConnectorScope::read("artifacts.read", "Read artifact metadata and bytes"),
        ConnectorScope::write("artifacts.publish", "Publish a new artifact receipt"),
    ];
    a.object_types = vec![
        ObjectType::new("artifact", "A registered artifact"),
        ObjectType::new("receipt", "A verification or publish receipt"),
        ObjectType::new("manifest", "An artifact manifest"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::LocalWatch;
    a.offline_cache = OfflineCache::metadata(64 * 1024 * 1024);
    a.rate_limit = RateLimit::local();
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::SecretAccess,
    ];
    a.revocation = RevocationPolicy::real_local();
    a.honesty_notes =
        "DECLARED only in this crate. Local artifact store wiring is a separate concern.".into();
    a
}

/// Every family ABI, stable order.
pub fn all_families() -> Vec<ConnectorAbi> {
    vec![
        local_folder(),
        rss(),
        github(),
        google_drive(),
        icloud_drive(),
        gmail(),
        google_calendar(),
        google_contacts(),
        slack(),
        notion(),
        dropbox_onedrive(),
        browser_search(),
        generic_mcp(),
        generic_oauth_api(),
        hawking_artifact_registry(),
    ]
}

/// Family ids that are implemented end-to-end.
pub fn implemented_family_ids() -> &'static [&'static str] {
    &["local_folder", "rss"]
}
