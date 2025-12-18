# Release v1.0.0

**Date: December 16, 2025**

We are thrilled to announce the first major release of the Stock Prices Dashboard, bringing interactive stock analysis right to your browser with Streamlit!

## ✨ New Features

This release introduces the core functionality of the application, allowing users to:

*   **View Historical Stock Prices**: Easily fetch and visualize historical price data for multiple tickers.
*   **Interactive Charting**: Explore price trends with customizable line and candlestick charts.
*   **Technical Indicator Overlays**: Enhance your analysis with built-in Moving Averages (MA), 52-week High/Low, and Relative Strength Index (RSI).
*   **Real-time News Headlines**: Stay informed with integrated news feeds for selected stocks.
*   **Flexible Ticker Selection**: Choose from a list of popular stocks or add your own custom tickers.
*   **Customizable Date Ranges**: Analyze specific periods by adjusting the start and end dates for data fetching.

## ⚙️ Under the Hood Improvements

To ensure reliability and maintainability, this release also includes significant improvements to the project's development practices:

*   **Comprehensive Unit Tests**: A new `tests/` directory with `pytest` unit tests has been added to validate the core logic of data processing and indicator calculations.
*   **Automated Continuous Integration (CI)**: A GitHub Actions workflow (`.github/workflows/ci.yml`) is now in place to automatically run tests on every push and pull request to the `main` branch, ensuring code quality and preventing regressions.

## 🚀 Get Started

To run the Streamlit Stock Prices Dashboard locally:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd streamlit-dashboard
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```

We hope you enjoy using the Stock Prices Dashboard! Feel free to report any issues or suggest new features.
