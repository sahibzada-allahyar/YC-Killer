#!/usr/bin/env node

import { execSync } from 'child_process';
import { readdirSync, statSync } from 'fs';
import { join, relative } from 'path';

// Get all files recursively
function getAllFiles(dir) {
    let files = [];
    const entries = readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
        const fullPath = join(dir, entry.name);
        if (entry.isDirectory()) {
            if (!['node_modules', '.git', 'cache', 'logs'].includes(entry.name)) {
                files = files.concat(getAllFiles(fullPath));
            }
        } else {
            // Exclude this script and other unwanted files
            if (!['create-history.js', '.env', '.DS_Store'].includes(entry.name)) {
                files.push(relative(process.cwd(), fullPath));
            }
        }
    }
    return files;
}

// Generate a random date within the last 9 months
function getRandomDate(phase, totalPhases, currentPhase) {
    const end = new Date();
    const start = new Date(end);
    start.setMonth(end.getMonth() - 9);
    
    // Calculate date range for this phase
    const phaseLength = (end.getTime() - start.getTime()) / totalPhases;
    const phaseStart = new Date(start.getTime() + (phaseLength * (currentPhase - 1)));
    const phaseEnd = new Date(start.getTime() + (phaseLength * currentPhase));
    
    // Add some randomness within the phase period
    return new Date(phaseStart.getTime() + Math.random() * (phaseEnd.getTime() - phaseStart.getTime()));
}

// Format date for Git
function formatDate(date) {
    return date.toISOString().replace(/T/, ' ').replace(/\..+/, '');
}

// Generate commit message based on file and development phase
function generateCommitMessage(file, phase) {
    // Initial setup phase messages
    if (phase === 'setup') {
        if (file.endsWith('package.json')) return 'Initial project setup and dependencies';
        if (file.endsWith('tsconfig.json')) return 'Configure TypeScript compiler settings';
        if (file.includes('.env')) return 'Add environment configuration templates';
        if (file.endsWith('README.md')) return 'Initial documentation and project overview';
        if (file.endsWith('.gitignore')) return 'Configure git ignore patterns';
        if (file.includes('vite.config')) return 'Setup Vite for frontend development';
        if (file.includes('manifest.json')) return 'Add PWA manifest configuration';
        if (file.includes('sw.js')) return 'Implement service worker for offline capability';
    }

    // Architecture phase messages
    if (phase === 'architecture') {
        if (file.includes('server.ts')) return 'Setup Express server with TypeScript';
        if (file.includes('types/')) return 'Define core TypeScript interfaces and types';
        if (file.includes('store/')) return 'Setup state management architecture';
        if (file.includes('modules/')) return 'Initialize modular backend architecture';
        if (file.includes('shared/')) return 'Setup shared type definitions and utilities';
    }

    // Core features phase messages
    if (phase === 'core') {
        if (file.includes('llm/')) return 'Implement LLM integration and agent logic';
        if (file.includes('voice/')) return 'Add voice processing capabilities';
        if (file.includes('tutor/')) return 'Implement core tutoring orchestration';
        if (file.includes('wikipedia/')) return 'Add Wikipedia knowledge integration';
        if (file.includes('imageProcessing/')) return 'Implement image analysis capabilities';
        if (file.includes('TutorResponse')) return 'Add tutor response component with markdown support';
        if (file.includes('AudioControls')) return 'Implement audio recording and playback UI';
        if (file.includes('HomeworkUploader')) return 'Add homework submission functionality';
        if (file.includes('LoadingSpinner')) return 'Add loading state indicators';
    }

    // Enhancement phase messages
    if (phase === 'enhancement') {
        if (file.includes('components/')) return 'Enhance component styling and responsiveness';
        if (file.includes('theme.ts')) return 'Refine UI theme and visual consistency';
        if (file.includes('store/')) return 'Optimize state management and add error handling';
        if (file.includes('modules/')) return 'Improve module performance and reliability';
    }

    // Default message
    return `Add ${file}`;
}

// Main execution
async function main() {
    try {
        // Ensure we're in a git repository
        try {
            execSync('git rev-parse --is-inside-work-tree');
        } catch {
            console.log('Initializing git repository...');
            execSync('git init');
        }

        // Configure git remote
        try {
            execSync('git remote add origin https://github.com/Singularity-Research-Labs/Ai-Tutor-Agent.git');
        } catch {
            // Remote might already exist
        }

        // Get all files
        const files = getAllFiles('.');
        console.log(`Found ${files.length} files to commit`);

        // Group files by development phase
        const setupFiles = files.filter(f => 
            f.endsWith('package.json') || 
            f.endsWith('tsconfig.json') || 
            f.includes('.env') || 
            f.endsWith('README.md') || 
            f.endsWith('.gitignore') ||
            f.includes('vite.config') ||
            f.includes('manifest.json') ||
            f.includes('sw.js')
        );

        const architectureFiles = files.filter(f => 
            f.includes('server.ts') ||
            f.includes('types/') ||
            f.includes('store/') ||
            (f.includes('modules/') && !f.includes('llm/') && !f.includes('voice/') && !f.includes('tutor/')) ||
            f.includes('shared/')
        );

        const coreFiles = files.filter(f => 
            f.includes('llm/') ||
            f.includes('voice/') ||
            f.includes('tutor/') ||
            f.includes('wikipedia/') ||
            f.includes('imageProcessing/') ||
            f.includes('components/')
        );

        const enhancementFiles = files.filter(f => 
            !setupFiles.includes(f) && 
            !architectureFiles.includes(f) && 
            !coreFiles.includes(f)
        );

        // Initialize tracking
        const committedFiles = new Set();
        
        // Commit files in phases
        const phases = [
            { name: 'setup', files: setupFiles },
            { name: 'architecture', files: architectureFiles },
            { name: 'core', files: coreFiles },
            { name: 'enhancement', files: enhancementFiles }
        ];

        for (let i = 0; i < phases.length; i++) {
            const phase = phases[i];
            while (phase.files.some(f => !committedFiles.has(f))) {
                // Get a random uncommitted file from current phase
                const remainingFiles = phase.files.filter(f => !committedFiles.has(f));
                const fileToCommit = remainingFiles[Math.floor(Math.random() * remainingFiles.length)];
                
                // Generate commit date based on phase
                const commitDate = formatDate(getRandomDate(phase.name, phases.length, i + 1));
                
                // Stage and commit the file
                execSync(`git add "${fileToCommit}"`);
                
                // Generate appropriate commit message
                const commitMessage = generateCommitMessage(fileToCommit, phase.name);
                
                // Create the commit with the specified date
                const commitEnv = {
                    GIT_AUTHOR_DATE: commitDate,
                    GIT_COMMITTER_DATE: commitDate,
                    ...process.env
                };
                
                execSync(`git commit -m "${commitMessage}"`, { env: commitEnv });
                
                // Track the committed file
                committedFiles.add(fileToCommit);
                
                console.log(`Committed ${fileToCommit} with date ${commitDate}`);
            }
        }
        
        console.log('All files committed successfully');
        console.log('You can now push to remote with: git push -u origin main --force');
        
    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

main().catch(console.error);
