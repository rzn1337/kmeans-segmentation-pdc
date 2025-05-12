#pragma once
// src/utils/logger.h
// Simple logging utility

#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <mutex>

// Log levels
enum class LogLevel {
    INFO,
    WARNING,
    ERROR,
    DEBUG
};

class Logger {
private:
    std::string logFilePath;
    std::ofstream logFile;
    std::mutex logMutex;
    
    // Convert log level to string
    std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::INFO:    return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR:   return "ERROR";
            case LogLevel::DEBUG:   return "DEBUG";
            default:                return "UNKNOWN";
        }
    }
    
    // Get current timestamp as string
    std::string getTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto nowTime = std::chrono::system_clock::to_time_t(now);
        
        std::tm timeInfo;
        
        #ifdef _WIN32
        localtime_s(&timeInfo, &nowTime);
        #else
        localtime_r(&nowTime, &timeInfo);
        #endif
        
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeInfo);
        
        return std::string(buffer);
    }
    
public:
    // Constructor
    Logger(const std::string& filePath = "") : logFilePath(filePath) {
        if (!logFilePath.empty()) {
            logFile.open(logFilePath, std::ios::out | std::ios::app);
            if (!logFile.is_open()) {
                std::cerr << "Failed to open log file: " << logFilePath << std::endl;
            }
        }
    }
    
    // Destructor
    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }
    
    // Log a message
    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(logMutex);
        
        std::string logMessage = getTimestamp() + " [" + levelToString(level) + "] " + message;
        
        // Always output to console
        std::cout << logMessage << std::endl;
        
        // If log file is available, write to it
        if (logFile.is_open()) {
            logFile << logMessage << std::endl;
            logFile.flush();
        }
    }
    
    // Log info message
    void info(const std::string& message) {
        log(LogLevel::INFO, message);
    }
    
    // Log warning message
    void warning(const std::string& message) {
        log(LogLevel::WARNING, message);
    }
    
    // Log error message
    void error(const std::string& message) {
        log(LogLevel::ERROR, message);
    }
    
    // Log debug message
    void debug(const std::string& message) {
        log(LogLevel::DEBUG, message);
    }
};