#pragma once
#include <string>
#include <ctime>

// Helper: Convert "YYYY-MM-DD" to UNIX timestamp (seconds)
time_t date_to_unix(const std::string& date_str);

std::string unix_to_datetime(time_t unix_timestamp);

// Parse ISO 8601 datetime string ("YYYY-MM-DDTHH:MM:SS") to std::tm
std::tm parse_iso_datetime(const std::string& dt);

// Format std::tm to ISO 8601 datetime string ("YYYY-MM-DDTHH:MM:SS")
std::string format_iso_datetime(const std::tm& tm);