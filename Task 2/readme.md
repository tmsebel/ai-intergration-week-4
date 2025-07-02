# Task 2: Automated Login Testing with Selenium

## Overview

This project demonstrates automated testing of a web application's login functionality using Python's `unittest` framework and Selenium WebDriver. The tests are designed to verify both valid and invalid login scenarios for the Power Learn Project Academy platform.

## Files

- [`login_test.py`](login_test.py): Standalone Python script for running login tests.
- [`login_test.ipynb`](login_test.ipynb): Jupyter notebook version of the login test, with output and error analysis.
- [`test_summary.markdown`](test_summary.markdown): Summary and analysis of AI-enhanced test coverage.

## How It Works

The test suite (`LoginTest`) performs the following:

1. **Setup**:  
   - Launches a headless Chrome browser session.
   - Navigates to the login page.

2. **Test Cases**:
   - **Valid Credentials**:  
     Attempts to log in with a valid username and password, then checks for success indicators (e.g., "Dashboard" or "Welcome" in the page source).
   - **Invalid Credentials**:  
     Attempts to log in with invalid credentials, then checks for error messages (e.g., "Invalid credentials" or "Error" in the page source).

3. **Result Reporting**:  
   - Prints the number of passed and failed tests, as well as success and failure rates.

## Example Output

The notebook output shows both tests failing due to Selenium not finding the login button using the provided XPath selector. This suggests either the element's locator is incorrect or the page structure has changed.

## Troubleshooting

- **Element Not Found**:  
  If you encounter "no such element" errors, inspect the login page to confirm the correct IDs or XPath for the username, password, and login button fields.
- **Headless Mode**:  
  The notebook uses headless Chrome for compatibility with automated environments. Remove the `--headless` option for debugging with a visible browser window.

## How to Run

### Python Script

```sh
python [login_test.py](http://_vscodecontentref_/0)
```
