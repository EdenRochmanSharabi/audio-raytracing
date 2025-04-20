# GitHub Upload Instructions

Follow these steps to upload your Audio Ray Tracing project to your GitHub account.

## Prerequisites

1. Make sure you have a GitHub account. If not, create one at [github.com](https://github.com/)
2. Ensure Git is installed on your system and properly configured
3. Have your GitHub username and personal access token ready (or SSH key configured)

## Setup GitHub Repository

1. Login to your GitHub account
2. Click the "+" icon in the top-right corner and select "New repository"
3. Enter a repository name (e.g., "audio-raytracing")
4. Add an optional description
5. Choose public or private visibility
6. Do not initialize with a README, .gitignore, or license since you already have these files
7. Click "Create repository"

## Push Your Local Repository to GitHub

GitHub will show instructions after repository creation. Here's how to push your existing repository:

### If Using HTTPS:

```bash
# Navigate to your project directory (if not already there)
cd /Users/edenrochman/Documents/Offline_projects/AudioTracing

# Add the GitHub repository as a remote
git remote add origin https://github.com/YourUsername/audio-raytracing.git

# Push your code to GitHub
git push -u origin master
```

### If Using SSH:

```bash
# Navigate to your project directory (if not already there)
cd /Users/edenrochman/Documents/Offline_projects/AudioTracing

# Add the GitHub repository as a remote
git remote add origin git@github.com:YourUsername/audio-raytracing.git

# Push your code to GitHub
git push -u origin master
```

## Verify Repository Upload

1. Refresh your GitHub repository page
2. You should see all your files and commit history

## Setting Up GitHub Pages (Optional)

If you want to create a project website:

1. Go to your repository on GitHub
2. Click "Settings"
3. Scroll down to "GitHub Pages" section
4. Select the branch you want to use (usually "master" or "main")
5. Click "Save"
6. GitHub will provide a URL where your site is published

## Additional GitHub Features to Consider

- **Issues**: Track bugs, enhancements, and other work items
- **Pull Requests**: Collaborate with others through code reviews
- **Actions**: Set up automated workflows for testing, deployment, etc.
- **Projects**: Organize work with project boards
- **Wiki**: Create additional documentation

## Keeping Your Repository Updated

After making local changes:

```bash
# Stage your changes
git add .

# Commit your changes
git commit -m "Description of changes"

# Push to GitHub
git push
``` 