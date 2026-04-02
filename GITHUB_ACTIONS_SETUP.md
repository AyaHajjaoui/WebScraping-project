# Web Scraping Project - GitHub Actions Setup

This project now uses **GitHub Actions** for automated weather data collection and cleaning, replacing the local APScheduler-based scheduler.

## 📁 Project Structure

```
.github/
  workflows/
    scrape-and-clean.yml  # Main automation workflow
    README.md             # Workflow documentation

processing/
  clean_data.py           # Data cleaning and combining
  
data/
  raw/                    # Raw scraped CSV files
  processed/              # Cleaned combined data
  
logs/                     # Execution logs
```

## 🚀 GitHub Actions Workflow

### What's Automated

The `.github/workflows/scrape-and-clean.yml` workflow:

1. **Runs on Schedule**: Daily at 2 AM UTC (configurable)
2. **Manual Trigger**: Can also be run manually from GitHub Actions tab
3. **Executes**:
   - Weather data scraping from multiple sources
   - Data cleaning and combination (`clean_data.py`)
   - Automatic commit of changes to repository
   - Stores processed data as artifacts

### Configuration

**To modify the schedule:**
1. Edit `.github/workflows/scrape-and-clean.yml`
2. Change the `cron` expression under `on.schedule`:
   ```yaml
   schedule:
     - cron: '0 2 * * *'  # Change these numbers
   ```
3. Common patterns:
   - `'0 2 * * *'` - 2 AM every day
   - `'0 */6 * * *'` - Every 6 hours
   - `'0 12 * * 0'` - Noon on Sundays
   - See [cron.guru](https://cron.guru) for help

### Manual Execution

1. Go to your repository on GitHub
2. Click **Actions** tab
3. Select **"Scrape Weather Data & Clean"** workflow
4. Click **"Run workflow"** button

### Monitoring

**View Results:**
- Go to **Actions** tab on GitHub
- Click on the workflow run to see logs
- Download data artifacts after completion

**Set Up Notifications:**
1. GitHub → Settings → Notifications
2. Enable workflow notifications

## 🔄 Migration from APScheduler

### What Changed

| Before | After |
|--------|-------|
| Local scheduler (APScheduler) | GitHub Actions |
| Runs continuously on your machine | Runs on GitHub servers |
| Manual start: `python main.py` | Automatic scheduling |
| Stops if machine is off | Works 24/7 |

### Local Development

**To test scraping locally:**
```bash
python main.py
```

This runs the scraper once and exits (perfect for GitHub Actions).

## 📦 Dependencies

**No longer needed:**
- `apscheduler` (removed)

**Still required:**
- `pandas` - Data processing
- `requests` - Web scraping
- `openpyxl` - Excel export

Install with:
```bash
pip install -r requirements.txt
```

## 🔐 Security Notes

- GitHub Actions uses GitHub-hosted runners (no sensitive data on your machine)
- Repository content is secure; only you control what runs
- Credentials/API keys can be stored in GitHub Secrets if needed

### Adding API Keys (if needed)

1. Go to **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Add your secret (e.g., `API_KEY`)
4. Use in workflow:
   ```yaml
   - name: Run scraper
     env:
      API_KEY: ${{ secrets.API_KEY }}
      run: python main.py
   ```

## ✅ Benefits

- ✓ Runs automatically on schedule
- ✓ No local machine required
- ✓ Free tier includes 2,000 action-minutes/month
- ✓ Built-in logging and monitoring
- ✓ Easy to share/collaborate
- ✓ Automatic data backups via GitHub

## 📊 Data Locations

- **Raw data**: `data/raw/` (not committed)
- **Processed data**: `data/processed/weather_data.csv` (committed)
- **Logs**: `logs/` (cleanup happens automatically)

## 🆘 Troubleshooting

**Workflow not running:**
- Check if it's enabled: Actions tab → workflow → enable if needed
- Verify schedule syntax on cron.guru

**Check logs:**
- Click on a failed run to see detailed error messages

**Manual trigger not working:**
- Ensure you have push access to the repository

## 📝 Next Steps

1. ✅ Commit all files to GitHub
2. ✅ Enable GitHub Actions in repository Settings
3. ✅ Test by manually running workflow once
4. ✅ Verify data is being collected
5. ✅ Set up notifications (optional)

---

For more details, see `.github/workflows/README.md`
