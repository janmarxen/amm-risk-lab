#pragma once
#include <string>
#include <ctime>

// Helper: Convert "YYYY-MM-DD" to UNIX timestamp (seconds)
time_t to_unix_timestamp(const std::string& date_str);

std::string unix_to_datetime(time_t unix_timestamp);