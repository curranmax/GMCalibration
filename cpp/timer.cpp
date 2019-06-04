
#include "timer.h"

#include <chrono>
#include <map>
#include <stdlib.h>
#include <string>

const std::string Timer::secure_preprocessing = "secure_preprocessing";
const std::string Timer::secure_su_request = "secure_su_request";

const std::string Timer::secure_write = "secure_write";

const std::string Timer::plaintext_split_preprocessing = "plaintext_split_preprocessing";
const std::string Timer::plaintext_grid_preprocessing = "plaintext_grid_preprocessing";

const std::string Timer::opt_pt_request = "opt_pt_request";
const std::string Timer::unopt_pt_request = "unopt_pt_request";

void Timer::start(const std::string& tag) {
	if(current_tag != "") {
		std::cerr << "Trying to start timer with mismatched tags: (" << tag << ", " << current_tag << ")" << std::endl;
		exit(0);
	}

	current_start = std::chrono::high_resolution_clock::now();
	current_tag = tag;
}

void Timer::end(const std::string& tag) {
	if(current_tag != tag) {
		std::cerr << "Trying to end timer with mismatched tags: (" << tag << ", " << current_tag << ")" << std::endl;
		exit(0);
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> elapsed = end - current_start;

	auto itr = avg_durations.find(tag);
	if(itr == avg_durations.end()) {
		avg_durations[tag] = std::make_pair(1, elapsed.count());
	} else {
		itr->second.first++;
		itr->second.second = itr->second.second * (double(itr->second.first - 1.0) / double(itr->second.first)) + elapsed.count() / double(itr->second.first);
	}

	current_tag = "";
}

float Timer::getAverageDuration(const std::string& tag) const {
	auto itr = avg_durations.find(tag);
	if(itr == avg_durations.cend()) {
		return 0.0;
	}
	return itr->second.second;
}

int Timer::numDurations(const std::string& tag) const {
	auto itr = avg_durations.find(tag);
	if(itr == avg_durations.cend()) {
		return 0;
	}

	return itr->second.first;
}
