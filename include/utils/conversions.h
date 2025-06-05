#pragma once

#include <vector>
#include <string>

/// Converts a vector of values from a given currency to USD.
/// @param values Vector of values in the original currency.
/// @param currency_symbol Symbol of the original currency (e.g., "ETH").
/// @param datetime Vector of datetime strings corresponding to each value.
/// @return Vector of values converted to USD.
std::vector<double> convert_to_usd(
    const std::vector<double>& values,
    const std::string& currency_symbol,
    const std::vector<std::string>& datetime,
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs
);
