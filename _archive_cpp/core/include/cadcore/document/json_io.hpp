#pragma once

#include "cadcore/document/document.hpp"

#include <string>

namespace cad {

/// Serialize document to a custom JSON schema (cad-document-v1).
[[nodiscard]] std::string document_to_json(const Document& doc);

/// Parse cad-document-v1 JSON. Returns false on parse/schema error.
bool document_from_json(const std::string& json, Document& out, std::string* error = nullptr);

bool save_document(const Document& doc, const std::string& path);
bool load_document(const std::string& path, Document& out, std::string* error = nullptr);

}  // namespace cad
