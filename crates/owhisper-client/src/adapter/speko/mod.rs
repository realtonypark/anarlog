#[cfg(feature = "local")]
mod batch;

use super::{LanguageQuality, LanguageSupport};

// https://docs.speko.ai/relay
pub(crate) const DEFAULT_API_BASE: &str = "https://router.speko.dev";

#[derive(Clone, Default)]
pub struct SpekoAdapter;

impl SpekoAdapter {
    // Speko routes per request across upstream models, so language support is
    // only known once it fails with `capability_unsupported`.
    pub fn language_support_batch(_languages: &[anlg_language::Language]) -> LanguageSupport {
        LanguageSupport::Supported {
            quality: LanguageQuality::NoData,
        }
    }
}
