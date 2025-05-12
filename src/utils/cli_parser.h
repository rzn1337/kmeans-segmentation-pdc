#pragma once
// src/utils/cli_parser.h
// Command line argument parser utility

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

class CLIParser {
private:
    struct Option {
        std::string name;
        std::string description;
        bool required;
        std::string defaultValue;
        
        Option(const std::string& name, const std::string& description, bool required, const std::string& defaultValue = "")
            : name(name), description(description), required(required), defaultValue(defaultValue) {}
    };
    
    std::vector<Option> options;
    std::unordered_map<std::string, std::string> parsedOptions;
    
public:
    // Add option to the parser
    void addOption(const std::string& name, const std::string& description, bool required, const std::string& defaultValue = "") {
        options.emplace_back(name, description, required, defaultValue);
    }
    
    // Parse command line arguments
    bool parse(int argc, char** argv) {
        // Initialize with default values
        for (const auto& option : options) {
            if (!option.defaultValue.empty()) {
                parsedOptions[option.name] = option.defaultValue;
            }
        }
        
        // Parse arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            // Handle option name
            if (arg[0] == '-') {
                if (i + 1 < argc) {
                    std::string value = argv[i + 1];
                    
                    // Skip values that look like options
                    if (value[0] != '-') {
                        parsedOptions[arg] = value;
                        i++; // Skip the value
                    } else {
                        // Flag option with no value
                        parsedOptions[arg] = "true";
                    }
                } else {
                    // Flag option at the end
                    parsedOptions[arg] = "true";
                }
            }
        }
        
        // Check required options
        bool valid = true;
        for (const auto& option : options) {
            if (option.required && parsedOptions.find(option.name) == parsedOptions.end()) {
                std::cerr << "Missing required option: " << option.name << " (" << option.description << ")" << std::endl;
                valid = false;
            }
        }
        
        return valid;
    }
    
    // Check if option exists
    bool hasOption(const std::string& name) const {
        return parsedOptions.find(name) != parsedOptions.end();
    }
    
    // Get option value
    std::string getValue(const std::string& name) const {
        auto it = parsedOptions.find(name);
        if (it != parsedOptions.end()) {
            return it->second;
        }
        
        // Return default value if available
        for (const auto& option : options) {
            if (option.name == name && !option.defaultValue.empty()) {
                return option.defaultValue;
            }
        }
        
        return "";
    }
    
    // Print usage information
    void printUsage(const std::string& programName) const {
        std::cout << "Usage: " << programName << " ";
        
        for (const auto& option : options) {
            if (option.required) {
                std::cout << option.name << " <value> ";
            }
        }
        
        std::cout << "[options]" << std::endl;
        std::cout << "Options:" << std::endl;
        
        for (const auto& option : options) {
            std::cout << "  " << option.name << "\t" << option.description;
            if (!option.defaultValue.empty()) {
                std::cout << " (default: " << option.defaultValue << ")";
            }
            if (option.required) {
                std::cout << " [REQUIRED]";
            }
            std::cout << std::endl;
        }
    }
};