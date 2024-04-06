# Agentic Hospital

A sophisticated multi-agent healthcare system developed by Sahibzada A at Singularity Research, an organization dedicated to democratizing AI through open source. This system leverages Large Language Models (LLMs) to create an adaptive collaboration framework for medical decision-making, where multiple AI agents work together to provide comprehensive healthcare advice. The aim is to provide top tier healthcare access to everyone across the globe.

## Project Structure

```
YC-Killer/
└── Agentic-Hospital/
    └── src/
        ├── agents/           # Specialist agent implementations
        ├── orchestrator/     # Collaboration coordination
        ├── services/        # Supporting services
        ├── types/           # TypeScript definitions
        └── frontend/        # Web interface
```

## Prerequisites

- Node.js >= 14
- npm >= 6
- Anthropic API key

## Getting Started

1. Navigate to the project directory:
   ```bash
   cd YC-Killer/Agentic-Hospital
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Add your ANTHROPIC_API_KEY to .env
   ```

4. Start development server:
   ```bash
   npm run dev
   ```

5. Launch frontend (optional):
   ```bash
   npm run serve-frontend
   ```

## Development

### Available Scripts
- `npm run build`: Compile TypeScript
- `npm run start`: Run production server
- `npm run dev`: Run development server
- `npm run serve-frontend`: Serve frontend interface

## Features

### Intelligent Query Analysis
- Automatic complexity assessment
- Dynamic team assembly based on query requirements
- Contextual understanding of patient history

### Adaptive Multi-Agent Framework
- Low complexity: Single primary care physician
- Moderate complexity: 2-3 relevant specialists
- High complexity: Full team collaboration with coordinated response

### Comprehensive Specialist Network
- Primary Care Agent: General health assessment and coordination
- Fitness Coach Agent: Exercise and physical activity guidance
- Lifestyle Coach Agent: Behavioral and habit modification
- Diabetes Specialist Agent: Diabetes management and monitoring
- Nutrition Agent: Dietary advice and meal planning
- Team Coordinator: Orchestrates multi-agent collaboration
- Moderator: Ensures quality and consistency of advice

### Insurance Integration
- Real-time coverage verification
- Historical approval rate analysis
- Pre-authorization requirement checks
- Annual limit tracking
- Specialist referral management

## Insurance Coverage Categories

- Preventive Care
- Diagnostic Tests
- Chronic Condition Management
- Specialist Consultations
- Medication Management
- Therapeutic Services
- Emergency Care

## Contributing

See the main repository's [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Sahibzada A  
Singularity Research  
[Your preferred contact method/information]

---

Made with ❤️ by Singularity Research
