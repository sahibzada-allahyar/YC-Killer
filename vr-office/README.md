# VR Office - Privacy-First Virtual Workspace

> **A virtual reality office environment designed to restore remote work trust without surveillance**

## 🎯 Problem Statement

Remote work has exploded, but so has employer paranoia about productivity. Traditional monitoring solutions resort to invasive surveillance—keystroke logging, screenshot capture, and activity tracking—that damages trust and morale.

**The core tension:** How do you provide presence awareness and collaboration tools without crossing into surveillance?

## 💡 Solution

VR Office creates an immersive virtual workspace with:
- **Spatial presence**: See who's "at their desk" in real-time
- **Proximity voice chat**: Talk naturally to nearby colleagues with distance-based audio falloff
- **In-VR screen sharing**: Share your work on virtual monitors without content capture
- **Status indicators**: Available, Focus Mode, Do Not Disturb

**What we DON'T do:**
- ❌ Keystroke logging
- ❌ Screenshot surveillance
- ❌ Content monitoring
- ❌ Productivity scoring based on activity

## 👥 Target Users

**Primary:** Remote teams of 4-16 people
- Engineering teams
- Design teams
- Product teams
- Cross-functional squads

**Use cases:**
- Daily standup in VR
- Pair programming sessions
- Design reviews with screen sharing
- "Walk over and ask a question" moments
- Casual water cooler conversations

## 🔑 Key Differentiator

### Privacy-First Architecture

**Aggregate metrics only, zero content capture:**
- Dashboard shows: "8 people online, 3 in focus mode, avg. session 4.2 hours"
- Dashboard NEVER shows: Screenshots, keystrokes, app usage, or content

**Data philosophy:**
- Presence heartbeats (every 30s)
- Voice/video streams (ephemeral, not recorded)
- Aggregate statistics (anonymized)
- No surveillance, no recordings, no content inspection

## 🏗️ Technology Stack

- **VR Client:** Unity + OpenXR + XR Interaction Toolkit
- **Networking:** Photon PUN 2 (rooms/state sync)
- **Voice/Video:** Agora SDK (proximity audio + screen share)
- **Backend:** Next.js API + Supabase (Postgres)
- **Auth:** Auth0 / OIDC
- **Admin Dashboard:** Next.js + React

## 🚀 MVP Scope (v0.5.0)

### Core Features
1. Multi-user VR office environment (up to 16 users per room)
2. Proximity voice chat with spatial audio and distance falloff
3. In-VR screen sharing to virtual desk monitors
4. Presence status system (Available/Focus/DND)
5. Admin dashboard with aggregate metrics only
6. Secure authentication and authorization

### Platform Support
- **PCVR only** (Meta Quest Link/Air XR, Valve Index, HTC Vive, etc.)
- Cross-platform: Windows, macOS, Linux
- OpenXR for broad headset compatibility

### Non-Goals for MVP
- ❌ Mobile VR / Quest standalone
- ❌ >16 concurrent users per room
- ❌ Real-time collaboration tools (whiteboards, 3D modeling)
- ❌ Video calls separate from screen share
- ❌ Content surveillance features

## 📋 Documentation

- **[MVP Scope](docs/mvp-scope.md)** - Detailed feature breakdown and non-goals
- **[Architecture](docs/architecture.md)** - System design and data flows
- **[Roadmap](docs/roadmap.md)** - Milestone-based development plan
- **[Success Metrics](docs/success-metrics.md)** - Quantifiable performance targets
- **[Risk Register](docs/risks.md)** - Known risks and mitigation strategies

## 🎯 Success Metrics

**Performance:**
- ≥72 FPS in 10-person room with 2×720p video streams
- Voice intelligible at 64 kbps with <16m falloff distance
- Screen share latency: first frame ≤2 seconds from start

**Privacy:**
- Zero content capture or surveillance features
- Admin dashboard shows only aggregate presence data

**Adoption:**
- 10+ teams actively using for daily standups
- 80%+ user satisfaction on "feels productive, not surveilled"

## 🗺️ Milestones

- **[0.1.0 Bootstrapping & Community](docs/roadmap.md#milestone-010-bootstrapping--community)** - Repo setup, CI/CD, contribution guides
- **[0.2.0 Core Multiuser VR](docs/roadmap.md#milestone-020-core-multiuser-vr)** - Networking, avatars, scene loading *(In Progress)*
- **[0.3.0 Voice + Screen Share](docs/roadmap.md#milestone-030-voice--screen-share)** - Agora integration, proximity audio
- **[0.4.0 Presence & Admin](docs/roadmap.md#milestone-040-presence--admin)** - Status system, metrics dashboard
- **[0.5.0 Hardening & Release](docs/roadmap.md#milestone-050-hardening--release)** - Security audit, deployment, testing
- **[0.6.0 Fusion Migration](docs/roadmap.md#milestone-060-fusion-migration)** *(Optional)* - Migrate from PUN 2 to Photon Fusion

## 🤝 Contributing

We welcome contributions! Please see:
- [Contribution Guide](CONTRIBUTING.md) - How to get started
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community standards
- [GitHub Issues](../../issues?q=is%3Aissue+is%3Aopen+label%3Avr-office) - Find tasks to work on

**Quick start:**
```bash
# Clone the repository
git clone https://github.com/sahibzada-allahyar/YC-Killer.git
cd YC-Killer/vr-office

# Follow setup instructions (coming in 0.1.0)
```

## 📈 Progress Tracking

**Current milestone:** 0.2.0 Core Multiuser VR (In Progress)

See the [full roadmap](docs/roadmap.md) for detailed progress and the [tracking issue](#) for weekly updates.

## 📄 License

[Add license information]

## 🔗 References

- [Unity XR Best Practices](https://docs.unity3d.com/Manual/xr_best_practices.html)
- [Photon PUN 2 Documentation](https://doc.photonengine.com/pun/current/getting-started/pun-intro)
- [Agora Unity SDK](https://docs.agora.io/en/video-calling/develop/get-started-sdk?platform=unity)
- [OpenXR Specification](https://www.khronos.org/openxr/)

---

**Questions?** Open an issue or join our Discord (link coming soon)
