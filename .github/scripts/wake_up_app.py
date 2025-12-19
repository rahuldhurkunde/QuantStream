from playwright.sync_api import sync_playwright
import time

APP_URL = "https://app-dashboard-dzcw6pagkmxzz39gzbqf2n.streamlit.app"

def run():
    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print(f"Navigating to {APP_URL}...")
        page.goto(APP_URL, timeout=60000)
        
        # Streamlit "sleeping" page often has a button like "Yes, get this app back up!"
        # We will look for buttons with "wake" or "yes" in the name/text
        try:
            print("Checking for wake up buttons...")
            # Locator for the specific Streamlit Cloud wake up button
            # Note: The text might change, but usually contains "Yes" or "Wake"
            # We try a few selectors
            
            # Wait a moment for the overlay to appear if it's there
            page.wait_for_load_state("networkidle")
            
            # Try specific text often found on Streamlit Cloud
            wake_btn = page.get_by_role("button", name="Yes, get this app back up!")
            
            if wake_btn.count() > 0 and wake_btn.is_visible():
                print("Found 'Yes, get this app back up!' button. Clicking...")
                wake_btn.click()
                print("Clicked. Waiting for app to load...")
                page.wait_for_timeout(10000) # Wait 10s for reload
            else:
                print("No 'wake up' button detected immediately.")

        except Exception as e:
            print(f"Error checking/clicking button: {e}")

        # Check title to verify we are likely on the app
        try:
            page.wait_for_load_state("networkidle", timeout=30000)
            title = page.title()
            print(f"Current Page Title: {title}")
        except Exception as e:
             print(f"Error getting title: {e}")

        browser.close()
        print("Done.")

if __name__ == "__main__":
    run()
