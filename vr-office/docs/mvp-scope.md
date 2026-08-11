# MVP Scope - VR Office v0.5.0

> **Minimum Viable Product Definition**: A privacy-first VR workspace that enables 4-16 person remote teams to collaborate with presence awareness, proximity voice, and screen sharing—without surveillance.

## 🎯 MVP Goals

**Primary objective:** Prove that virtual presence can reduce remote work paranoia without invasive monitoring.

**Success criteria:**
- 10+ teams use VR Office for daily standups
- Users report feeling "productive, not surveilled" (≥80% satisfaction)
- Technical performance meets [success metrics](success-metrics.md)

## ✅ In Scope for MVP

### 1. Multi-User VR Office Environment

**Description:** A shared 3D office space where remote team members can see each other's presence.

**Features:**
- Unity-based VR environment with desks, meeting areas, and casual spaces
- OpenXR support for broad headset compatibility (Quest Link/Air XR, Index, Vive, etc.)
- XR Interaction Toolkit for locomotion (teleport, smooth movement)
- Persistent room IDs (team workspaces)
- Up to 16 concurrent users per room

**Technical stack:**
- Unity 2022.3 LTS or newer
- OpenXR + XR Interaction Toolkit
- Photon PUN 2 for room management and state synchronization

**Acceptance criteria:**
- [ ] 10 users can join the same room simultaneously
- [ ] Room persists across sessions (team can rejoin same space)
- [ ] Users can navigate the environment with teleport/smooth locomotion
- [ ] Performance: ≥72 FPS with 10 users in room (see [success metrics](success-metrics.md))

---

### 2. Proximity Voice Chat with Distance Falloff

**Description:** Spatial audio that mimics real-world conversations—talk to people near you, hear voices fade with distance.

**Features:**
- Proximity-based voice chat (hear colleagues within ~16m)
- Distance-based volume falloff (louder when close, quieter far away)
- Directional audio (voice comes from avatar's position)
- Mute/unmute controls
- Push-to-talk option

**Technical stack:**
- Agora Voice SDK or Photon Voice 2
- Unity Audio Source with spatial blend and distance curves

**Acceptance criteria:**
- [ ] Voice is intelligible at 64 kbps bitrate
- [ ] Audio falloff works smoothly from 0-16m
- [ ] Latency: <150ms for voice transmission
- [ ] Users can mute/unmute themselves
- [ ] Voice positioning matches avatar location

---

### 3. In-VR Screen Sharing to Desk Monitors

**Description:** Share your desktop screen to a virtual monitor in the VR environment—no content leaves the room.

**Features:**
- Share entire desktop or specific application window
- Screen appears on virtual monitor at user's desk
- Other users can walk up and view the shared screen
- Stop sharing at any time
- Maximum 2 concurrent screen shares per room (MVP limit)

**Technical stack:**
- Agora Video SDK for screen capture and streaming
- Unity RenderTexture for displaying video streams on 3D monitors

**Acceptance criteria:**
- [ ] Users can share desktop screen to virtual monitor
- [ ] Shared screen visible to other users in VR at ≥720p resolution
- [ ] Latency: First frame visible ≤2 seconds from clicking "Share"
- [ ] Screen sharing stops cleanly when user disconnects
- [ ] Performance: 2×720p streams with ≥72 FPS in 10-person room

---

### 4. Presence Status System

**Description:** Simple status indicators to show availability without content surveillance.

**Status options:**
- 🟢 **Available** - Open for conversation
- 🟡 **Focus Mode** - Working on task, minimize interruptions
- 🔴 **Do Not Disturb** - Important work, please don't interrupt

**Features:**
- Status badge visible above avatar
- Keyboard shortcut to change status (e.g., Ctrl+1/2/3)
- In-VR menu to set status
- Status persists until manually changed

**Technical stack:**
- Photon PUN 2 custom properties for status sync
- Unity UI for status badges

**Acceptance criteria:**
- [ ] Users can set status via in-VR menu or keyboard shortcut
- [ ] Status updates visible to all users in room within 2 seconds
- [ ] Status persists across room rejoin
- [ ] Clear visual differentiation between Available/Focus/DND

---

### 5. Admin Dashboard with Aggregate Metrics

**Description:** Web-based dashboard for team admins to see presence patterns—without surveillance.

**Metrics displayed (aggregates only):**
- Current users online
- Status distribution (e.g., "5 Available, 2 Focus, 1 DND")
- Average session duration (daily/weekly)
- Peak usage times (heatmap)
- Total room usage hours

**What is NOT displayed:**
- ❌ Individual user activity timelines
- ❌ Screenshots or content
- ❌ Productivity scores
- ❌ "Time away from desk"
- ❌ Application usage or keystrokes

**Technical stack:**
- Next.js for frontend
- Supabase (Postgres) for data storage
- Backend API (Node.js/Express or Next.js API routes)
- Real-time presence updates via WebSockets or polling

**Acceptance criteria:**
- [ ] Dashboard shows real-time user count and status distribution
- [ ] Charts display aggregate usage patterns (daily/weekly)
- [ ] No individual surveillance data exposed
- [ ] Dashboard loads in <2 seconds
- [ ] Mobile-responsive design

---

### 6. Authentication & Authorization

**Description:** Secure login and team-based access control.

**Features:**
- SSO/OIDC authentication (Auth0, Okta, or similar)
- Team workspace isolation (users only see their team's room)
- Admin role for dashboard access
- Member role for VR office access
- Invite system for adding team members

**Technical stack:**
- Auth0 or alternative OIDC provider
- JWT tokens for session management
- Supabase Auth integration

**Acceptance criteria:**
- [ ] Users can log in with SSO (Google, Microsoft, etc.)
- [ ] Team workspaces are isolated (no cross-team visibility)
- [ ] Admins can invite new members via email
- [ ] Sessions expire after 7 days (refresh token flow)
- [ ] Secure token storage (no plaintext credentials)

---

## ❌ Explicitly Out of Scope for MVP

### Platform & Scale
- **Mobile VR / Quest Standalone:** PCVR only for MVP (Quest Link/Air XR supported)
- **>16 concurrent users per room:** Hard limit at 16 for MVP
- **Multi-room navigation:** Each team has one persistent room

### Features
- **Video calls separate from screen share:** Screen share only; no face-to-face video
- **Real-time collaboration tools:** No whiteboards, 3D model editing, or co-editing
- **Persistent chat/messaging:** No text chat in MVP
- **File sharing/storage:** Use existing tools (Slack, Google Drive)
- **Calendar integration:** No meeting scheduling in MVP
- **Recording/playback:** No session recording whatsoever
- **Avatar customization:** Basic avatars only (no custom skins/accessories)
- **Hand tracking:** Controller-based interaction only

### Surveillance/Monitoring (Never in Scope)
- **Keystroke logging:** Explicit anti-feature
- **Screenshot capture:** Never implemented
- **Content monitoring:** No analysis of shared screens
- **Productivity scoring:** No individual performance metrics
- **Activity tracking:** No mouse movement, app usage, or idle detection beyond presence heartbeat

### Advanced Tech
- **Photon Fusion migration:** Deferred to v0.6.0 (optional post-MVP)
- **Self-hosted Photon/Agora:** Cloud services only for MVP
- **Custom voice codec:** Use Agora's default Opus codec
- **Lip sync/facial animation:** Not in MVP

---

## 🎯 MVP Milestone Alignment

This scope maps to the following milestones:

- **0.1.0 Bootstrapping & Community** → Repository setup, CI/CD, docs
- **0.2.0 Core Multiuser VR** → Environment, avatars, Photon networking
- **0.3.0 Voice + Screen Share** → Agora integration, proximity audio
- **0.4.0 Presence & Admin** → Status system, dashboard
- **0.5.0 Hardening & Release** → Security audit, deployment, MVP release

See [roadmap.md](roadmap.md) for detailed milestone breakdown.

---

## 📊 MVP Sizing

**Estimated effort:** 80-100 issues across 5 milestones (0.1.0-0.5.0)

**Timeline:** Target 12-16 weeks for contributor-driven development (open source)

**Team size:** 3-5 core contributors + community PRs

---

## 🔄 Post-MVP Considerations (v0.6.0+)

Features under consideration after MVP ships:

- Mobile VR support (Quest standalone)
- In-VR whiteboard/sticky notes
- Persistent text chat
- Avatar customization
- Multi-room office layouts
- Calendar integration for meetings
- Photon Fusion migration for improved performance

These will be scoped separately based on MVP feedback and adoption.

---

## ✅ Definition of Done

The MVP is complete when:

1. All features in "In Scope" section meet acceptance criteria
2. All items in "Out of Scope" are documented and deferred/rejected
3. [Success metrics](success-metrics.md) are met
4. Security audit passed (no critical/high vulnerabilities)
5. 3+ teams actively using VR Office for daily standups
6. Documentation complete (setup guides, API docs, user manual)

---

**Questions or feedback?** Comment on the [tracking issue](../../issues) or open a discussion.
