#include "utils/subgraph_utils.h"
#include "utils/time_utils.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <iostream>
#include <fstream>
#include <algorithm>

using json = nlohmann::json;

size_t WriteCallback_subgraph_utils(void* contents, size_t size, size_t nmemb, std::string* buffer) {
    size_t totalSize = size * nmemb;
    buffer->append((char*)contents, totalSize);
    return totalSize;
}

std::optional<json> graphql_post(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& graphqlQuery
) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    json payload = {
        {"query", graphqlQuery},
        {"operationName", "Subgraphs"},
        {"variables", {}}
    };
    std::string postData = payload.dump();

    curl = curl_easy_init();
    if (curl) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        std::string authHeader = "Authorization: Bearer " + apiSubgraphs;
        headers = curl_slist_append(headers, authHeader.c_str());

        std::string url = "https://gateway.thegraph.com/api/subgraphs/id/" + idSubgraphs;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback_subgraph_utils);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);

        if (res != CURLE_OK || http_code != 200) {
            std::cerr << "cURL error: " << curl_easy_strerror(res) << std::endl;
            return std::nullopt;
        }

        try {
            return json::parse(readBuffer);
        } catch (const std::exception& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
        }
    }
    return std::nullopt;
}

std::optional<PriceDatetimeSeries> parse_price_datetime_series(const json& json_data) {
    try {
        PriceDatetimeSeries result;
        std::string token0_symbol, token1_symbol;
        if (json_data.contains("data") && json_data["data"].contains("pool")) {
            const auto& pool = json_data["data"]["pool"];
            token0_symbol = pool["token0"]["symbol"].get<std::string>();
            token1_symbol = pool["token1"]["symbol"].get<std::string>();
        }
        if (json_data.contains("data") && json_data["data"].contains("poolHourDatas")) {
            for (const auto& hour_data : json_data["data"]["poolHourDatas"]) {
                time_t timestamp = hour_data["periodStartUnix"].get<time_t>();
                result.datetime.push_back(unix_to_datetime(timestamp));
                result.price.push_back(std::stod(hour_data["token0Price"].get<std::string>()));
            }
        }
        // Optionally, you could store token0_symbol/token1_symbol in result if needed
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing price/datetime series: " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::optional<PriceDatetimeSeries> fetch_price_datetime_series(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& poolAddress,
    const std::string& startDate,
    const std::string& endDate
) {
    time_t start_ts = date_to_unix(startDate);
    time_t end_ts = date_to_unix(endDate) + 86399;

    // Query both pool info (for token0/token1 symbols) and poolHourDatas
    std::string graphqlQuery = R"({
    pool(id: ")" + poolAddress + R"(") {
        token0 { symbol }
        token1 { symbol }
    }
    poolHourDatas(
        orderBy: periodStartUnix,
        orderDirection: asc,
        where: { pool: ")" + poolAddress + R"(", periodStartUnix_gte: )" + std::to_string(start_ts) + R"(, periodStartUnix_lte: )" + std::to_string(end_ts) + R"( }
    ) {
        periodStartUnix
        token0Price
    }
    })";

    auto json_data_opt = graphql_post(apiSubgraphs, idSubgraphs, graphqlQuery);
    if (!json_data_opt.has_value()) return std::nullopt;
    return parse_price_datetime_series(json_data_opt.value());
}

// Helper to load pool addresses from JSON file
std::unordered_map<std::string, std::unordered_map<std::string, std::string>> load_pool_addresses(const std::string& json_path) {
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> pool_map;
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] Could not open pool addresses JSON file: " << json_path << std::endl;
        std::string abs_path = "../" + json_path;
        file.open(abs_path);
        if (!file.is_open()) {
            std::cerr << "[DEBUG] Could not open fallback pool addresses JSON file: " << abs_path << std::endl;
            std::string just_filename = "v3_pool_addresses.json";
            file.open(just_filename);
            if (!file.is_open()) {
                std::cerr << "[DEBUG] Could not open fallback pool addresses JSON file: " << just_filename << std::endl;
                return pool_map;
            }
        }
    }
    nlohmann::json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "[DEBUG] Failed to parse JSON: " << e.what() << std::endl;
        return pool_map;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string subgraph_id = it.key();
        std::transform(subgraph_id.begin(), subgraph_id.end(), subgraph_id.begin(), ::tolower);
        if (it->is_object()) {
            for (auto pit = it->begin(); pit != it->end(); ++pit) {
                std::string pool_key = pit.key();
                std::transform(pool_key.begin(), pool_key.end(), pool_key.begin(), ::toupper);
                std::string address = pit.value();
                std::transform(address.begin(), address.end(), address.begin(), ::tolower);
                pool_map[subgraph_id][pool_key] = address;
            }
        }
    }
    return pool_map;
}

// Helper to get pool address for a currency symbol pair (e.g., "ETH_USDC" or "USDC_ETH") and subgraph id
std::string get_pool_address(const std::string& symbol_pair, const std::string& subgraph_id) {
    static auto pool_map = load_pool_addresses("utils/v3_pool_addresses.json");
    // Normalize to uppercase for pool key and lowercase for subgraph id
    std::string key = symbol_pair;
    std::transform(key.begin(), key.end(), key.begin(), ::toupper);
    std::string sub_id = subgraph_id;
    std::transform(sub_id.begin(), sub_id.end(), sub_id.begin(), ::tolower);
    auto sub_it = pool_map.find(sub_id);
    if (sub_it == pool_map.end()) {
        return "";
    }
    auto& pools = sub_it->second;
    auto it = pools.find(key);
    if (it != pools.end()) {
        return it->second;
    }
    // Try reversed pair in uppercase
    size_t sep = key.find('_');
    if (sep != std::string::npos) {
        std::string reversed = key.substr(sep + 1) + "_" + key.substr(0, sep);
        it = pools.find(reversed);
        if (it != pools.end()) {
            return it->second;
        }
    }
    return "";
}