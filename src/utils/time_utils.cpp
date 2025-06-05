#include "utils/time_utils.h"
#include <cstring>
#include <sstream>
#include <iomanip>

time_t date_to_unix(const std::string& date_str) {
    struct tm tm_time = {};
    // Parse date string, assuming format "YYYY-MM-DD"
    strptime(date_str.c_str(), "%Y-%m-%d", &tm_time);
    // Set time to start of day UTC
    tm_time.tm_hour = 0;
    tm_time.tm_min = 0;
    tm_time.tm_sec = 0;
    // Use timegm to get UTC timestamp (if available)
    return timegm(&tm_time);
}

// Helper function to convert UNIX timestamp to datetime string
std::string unix_to_datetime(time_t unix_timestamp) {
    std::tm* tm = std::localtime(&unix_timestamp);
    std::stringstream ss;
    ss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::tm parse_iso_datetime(const std::string& dt) {
    std::tm tm = {};
    std::istringstream ss(dt);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse datetime: " + dt);
    }
    return tm;
}

std::string format_iso_datetime(const std::tm& tm) {
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}