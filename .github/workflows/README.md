# GitHub Actions Workflows

This directory contains automated workflows for the Web Scraping project.

## Workflows

### scrape-and-clean.yml
**Automated Weather Data Collection & Cleaning**

- **Schedule**: Daily at 2 AM UTC (configurable in cron expression)
- **Trigger**: Can also be manually triggered via GitHub Actions tab

**What it does:**
1. Checks out the repository
2. Sets up Python environment
3. Installs dependencies (pandas, requests, openpyxl)
4. Runs the weather scraper (`main.py`)
5. Cleans and combines raw data (`processing/clean_data.py`)
6. Commits changes to repository
7. Uploads processed data as artifacts

**Configuration:**
- To change schedule frequency, modify the `cron` expression in `on.schedule`
- Cron format: `'minute hour day month weekday'`
- Common examples:
  - `'0 2 * * *'` - 2 AM every day (UTC)
  - `'0 */6 * * *'` - Every 6 hours
  - `'0 12 * * 0'` - Noon every Sunday

## Environment Setup

The workflow uses GitHub-provided runners with:
- Ubuntu latest
- Python 3.12
- Dependencies installed from requirements

## Authentication

For API scraping (if needed):
- Store API keys in GitHub Secrets
- Add to workflow env or pass as environment variables

Example:
```yaml
- name: Run scraper with API key
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: python main.py
```

## Monitoring

- View workflow runs in the **Actions** tab on GitHub
- Check logs for each step
- Download artifacts (processed data, logs) after completion
- Set up notifications for workflow failures

## Manual Trigger

Click the **Run workflow** button in the Actions tab to manually start a scrape at any time.
