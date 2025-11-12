# Firebase Cloud Functions: Complete User Manual for AI Agents

## Purpose

This manual teaches AI agents how to build serverless cloud functions using Firebase. It covers **70+ official examples**, best practices, and production patterns.

**What you'll learn**:
- Event-driven and request-response functions
- Firebase service integrations (Auth, Storage, Database, Messaging)
- External service patterns (APIs, webhooks, payments)
- Testing, security, and observability

**Repository**: https://github.com/firebase/functions-samples

---

## Quick Start (5 Minutes)

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login to Firebase
firebase login

# Navigate to example
cd functions-samples/Node/https-time-server

# Install dependencies
cd functions && npm install && cd ..

# Run locally
firebase emulators:start

# Deploy to production
firebase deploy --only functions
```

**Expected Result**: HTTP endpoint at `http://localhost:5001` returning formatted time

---

## Core Concepts

### 1. Trigger Types

Firebase Cloud Functions execute in response to events. There are three categories:

#### Event-Driven Triggers (Asynchronous)
Functions execute automatically when events occur in Firebase or Google Cloud:

| Trigger | When It Fires | Example Use Case |
|---------|---------------|-------------------|
| **Firestore** | Document created/updated/deleted | Validate data, send notifications |
| **Realtime Database** | Node value changes | Real-time chat, presence tracking |
| **Cloud Storage** | File uploaded/deleted | Generate thumbnails, scan for malware |
| **Pub/Sub** | Message published to topic | Decouple microservices, fan-out work |
| **Firebase Auth** | User created/signed in | Block suspicious accounts, welcome emails |
| **Custom Events** | Extension publishes event | React to third-party integrations |

#### Request-Response Triggers (Synchronous)
Functions respond to direct calls and wait for a response:

| Trigger | How It's Called | Example Use Case |
|---------|-----------------|-------------------|
| **HTTPS** | HTTP request (GET, POST, etc.) | REST APIs, webhooks |
| **Callable** | Firebase SDK `.call()` | Client-server RPC, authenticated operations |
| **Streaming** | Callable with streaming | Long responses, real-time data |

#### Scheduled Triggers
Functions run on a schedule:

| Trigger | Configuration | Example Use Case |
|---------|--------------|-------------------|
| **Cloud Scheduler** | Cron expression | Daily cleanup, weekly reports |

#### Task Queue Triggers
Functions process distributed work:

| Trigger | Configuration | Example Use Case |
|---------|--------------|-------------------|
| **Task Queue** | Rate limits, retries | API rate limiting, batch processing |

### 2. Execution Model

**Node.js (2nd Generation - Recommended)**:
- Modern Node.js runtime (18+)
- Better cold start performance
- More granular scaling
- 70% of examples use this

**Python (2nd Generation - Preview)**:
- Python 3.8+ runtime
- Same trigger types as Node.js
- Growing ecosystem
- 16% of examples use this

**1st Generation (Legacy)**:
- Node.js 10-16
- Still widely used (44 examples)
- Migration path to 2nd gen available

---

## Learning Path (8 Weeks)

### Week 1-2: Fundamentals

**Goal**: Understand basic triggers and Firebase Admin SDK

**Quickstarts to Study**:

1. **https-time-server** (Node & Python)
   - 15 minutes
   - HTTP endpoint returning formatted time
   - Learn: Request/response pattern, parameter handling

```javascript
const {onRequest} = require("firebase-functions/https");
const moment = require("moment");

exports.date = onRequest((req, res) => {
  const format = req.query.format || req.body?.format;
  res.status(200).send(moment().format(format));
});
```

2. **uppercase-firestore** (Node & Python)
   - 20 minutes
   - Listen for new Firestore documents, uppercase a field
   - Learn: Event triggers, Admin SDK writes

```javascript
const {onDocumentCreated} = require("firebase-functions/firestore");

exports.makeuppercase = onDocumentCreated("/messages/{docId}", (event) => {
  const original = event.data.data().original;
  return event.data.ref.set({uppercase: original.toUpperCase()}, {merge: true});
});
```

3. **uppercase-rtdb** (Node & Python)
   - 20 minutes
   - Realtime Database trigger with data transformation
   - Learn: Database references, path parameters

**Knowledge Check**:
- What's the difference between event-driven and request-response?
- How do you access `event.data` vs `event.params`?
- When should you return a promise?

### Week 2-3: Client Communication

**Goal**: Build authenticated client-server interactions

**Quickstart to Study**:

**callable-functions** (Node & Python)
- 30 minutes
- Client SDK communication with authentication
- Learn: Auth context, input validation, error handling

```javascript
const {onCall, HttpsError} = require("firebase-functions/https");

exports.addnumbers = onCall((request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Must be logged in");
  }

  const {firstNumber, secondNumber} = request.data;
  if (!Number.isFinite(firstNumber) || !Number.isFinite(secondNumber)) {
    throw new HttpsError("invalid-argument", "Arguments must be numbers");
  }

  return {result: firstNumber + secondNumber};
});
```

**Key Pattern**: Always validate input and check authentication

**Knowledge Check**:
- When to use `HttpsError` vs throwing regular errors?
- How to access `request.auth.uid`?
- Difference between callable and HTTPS endpoints?

### Week 3-4: File Processing

**Goal**: Handle asynchronous file operations

**Quickstart to Study**:

**thumbnails** (Node & Python)
- 45 minutes
- Generate image thumbnails on upload
- Learn: Storage triggers, external libraries, conditional logic

```javascript
const {onObjectFinalized} = require("firebase-functions/storage");
const sharp = require("sharp");
const path = require("path");

exports.generateThumbnail = onObjectFinalized(async (event) => {
  const filePath = event.data.name;
  const contentType = event.data.contentType;

  // Filter: Skip non-images and already-processed thumbnails
  if (!contentType.startsWith("image/")) return;
  if (path.basename(filePath).startsWith("thumb_")) return;

  // Download, process, upload
  const bucket = getStorage().bucket(event.data.bucket);
  const [imageBuffer] = await bucket.file(filePath).download();
  const thumbnail = await sharp(imageBuffer).resize(200, 200).toBuffer();

  const thumbPath = path.join(path.dirname(filePath), `thumb_${path.basename(filePath)}`);
  await bucket.file(thumbPath).save(thumbnail);
});
```

**Knowledge Check**:
- How to avoid infinite loops (processing thumbnails of thumbnails)?
- When to allocate more CPU/memory?
- Error handling for file operations?

### Week 4-5: Security & Validation

**Goal**: Control access and enforce business rules

**Quickstart to Study**:

**auth-blocking-functions** (Node & Python)
- 40 minutes
- Block user registration/login based on rules
- Learn: Auth event blocking, database lookups, rejection patterns

```javascript
const {beforeUserCreated, beforeUserSignedIn, HttpsError} = require("firebase-functions/identity");
const {getFirestore} = require("firebase-admin/firestore");

// Only allow company emails
exports.validatenewuser = beforeUserCreated((event) => {
  if (!event.data.email?.includes("@acme.com")) {
    throw new HttpsError("invalid-argument", "Only acme.com emails allowed");
  }
});

// Check banned list
exports.checkforban = beforeUserSignedIn(async (event) => {
  const email = event.data.email;
  const doc = await getFirestore().collection("banned").doc(email).get();
  if (doc.exists) {
    throw new HttpsError("permission-denied", "Account banned");
  }
});
```

**Knowledge Check**:
- Difference between `beforeUserCreated` and `beforeUserSignedIn`?
- When to perform database lookups in auth flow?
- Impact on user experience?

### Week 5-6: Asynchronous Processing

**Goal**: Handle long-running operations and external APIs

**Quickstarts to Study**:

1. **pubsub-helloworld** (Node & Python)
   - 25 minutes
   - Process Pub/Sub messages
   - Learn: Message parsing, base64 decoding

2. **taskqueues-backup-images** (Node & Python)
   - 1 hour
   - Rate-limited external API integration
   - Learn: Task queues, retries, secret management

```javascript
const {onTaskDispatched} = require("firebase-functions/tasks");

exports.backupapod = onTaskDispatched({
  retryConfig: {maxAttempts: 5, minBackoffSeconds: 60},
  rateLimits: {maxConcurrentDispatches: 6},  // Max 6 concurrent
}, async (req) => {
  const date = req.body.date;

  // Call NASA API (rate-limited externally)
  const response = await fetch(`https://api.nasa.gov/planetary/apod?api_key=${SECRET}&date=${date}`);
  const data = await response.json();

  // Save to Firestore
  await getFirestore().collection("apod").doc(date).set(data);
});

// Enqueue task with delay
const queue = functions.getQueue("backupapod");
queue.enqueue({date: "2023-01-01"}, {scheduleDelaySeconds: 60});
```

**Knowledge Check**:
- Why use Task Queues instead of direct API calls?
- How do retry policies work?
- Concurrency control patterns?

### Week 6-7: Real-Time Notifications

**Goal**: Send push notifications with Firebase Cloud Messaging

**Quickstart to Study**:

**fcm-notifications** (Node)
- 1 hour
- Send notifications when user is followed
- Learn: FCM, batch operations, token management

```javascript
const {onValueWritten} = require("firebase-functions/database");
const {getMessaging} = require("firebase-admin/messaging");

exports.sendFollowerNotification = onValueWritten("/followers/{followedUid}/{followerUid}", async (event) => {
  const followedUid = event.params.followedUid;
  const followerUid = event.params.followerUid;

  // Get follower info
  const followerSnap = await admin.database().ref(`/users/${followerUid}`).once("value");
  const followerName = followerSnap.val()?.name || "Someone";

  // Get followed user's device tokens
  const tokensSnap = await admin.database().ref(`/users/${followedUid}/tokens`).once("value");
  const tokens = Object.keys(tokensSnap.val() || {});

  // Send batch notifications
  const messages = tokens.map(token => ({
    token,
    notification: {title: "New Follower", body: `${followerName} followed you`},
  }));

  const response = await getMessaging().sendEach(messages);
  console.log(`Sent ${response.successCount} notifications`);
});
```

**Knowledge Check**:
- How to manage device tokens?
- Batch sending patterns?
- Error handling for invalid tokens?

### Week 7: Testing

**Goal**: Write testable, production-ready functions

**Quickstarts to Study**:

1. **test-functions-jest** (Node)
   - 45 minutes
   - Unit testing with Jest
   - Learn: Mocking, test structure

2. **test-functions-mocha** (Node)
   - 45 minutes
   - Alternative testing with Mocha + Chai

```javascript
// tests/index.test.js
const test = require("firebase-functions-test")();
const myFunctions = require("../index");

describe("addnumbers", () => {
  it("adds two numbers", () => {
    const result = myFunctions.addnumbers({
      data: {firstNumber: 2, secondNumber: 3},
      auth: {uid: "test-user"},
    });

    expect(result.result).toBe(5);
  });

  it("rejects unauthenticated requests", () => {
    expect(() => {
      myFunctions.addnumbers({data: {firstNumber: 2, secondNumber: 3}});
    }).toThrow("unauthenticated");
  });
});
```

**Knowledge Check**:
- How to mock Admin SDK?
- Unit vs integration tests?
- Testing async triggers?

### Week 8: Production Patterns

**Goal**: Build production-grade functions

**Quickstarts to Study**:

1. **delete-unused-accounts-cron** (Node & Python)
   - 45 minutes
   - Scheduled user cleanup
   - Learn: Cloud Scheduler, bulk operations

2. **instrument-with-opentelemetry** (Node)
   - 1 hour
   - Distributed tracing
   - Learn: Observability, custom spans

3. **callable-functions-streaming** (Node)
   - 45 minutes
   - Stream response data
   - Learn: Real-time streaming

```javascript
// Scheduled cleanup
exports.deleteInactiveUsers = onSchedule("every day 00:00", async () => {
  const inactiveDate = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000); // 90 days ago

  const listResult = await getAuth().listUsers(1000);
  const promises = listResult.users
    .filter(user => new Date(user.metadata.lastSignInTime) < inactiveDate)
    .map(user => getAuth().deleteUser(user.uid));

  await Promise.all(promises);
  console.log(`Deleted ${promises.length} inactive users`);
});
```

**Knowledge Check**:
- Cron expression syntax?
- Bulk operation efficiency?
- OpenTelemetry setup?

---

## Trigger Syntax Reference

### Node.js 2nd Gen

```javascript
// Firestore
const {onDocumentCreated, onDocumentUpdated, onDocumentDeleted, onDocumentWritten} = require("firebase-functions/firestore");
exports.myTrigger = onDocumentCreated("/path/{id}", (event) => {/*...*/});

// Realtime Database
const {onValueCreated, onValueUpdated, onValueDeleted, onValueWritten} = require("firebase-functions/database");
exports.myTrigger = onValueCreated("/path/{id}", (event) => {/*...*/});

// Storage
const {onObjectFinalized, onObjectDeleted} = require("firebase-functions/storage");
exports.myTrigger = onObjectFinalized((event) => {/*...*/});

// Pub/Sub
const {onMessagePublished} = require("firebase-functions/pubsub");
exports.myTrigger = onMessagePublished("topic-name", (event) => {/*...*/});

// Auth
const {beforeUserCreated, beforeUserSignedIn} = require("firebase-functions/identity");
exports.myTrigger = beforeUserCreated((event) => {/*...*/});

// HTTPS
const {onRequest, onCall} = require("firebase-functions/https");
exports.myEndpoint = onRequest((req, res) => {/*...*/});
exports.myCallable = onCall((request) => {/*...*/});

// Task Queue
const {onTaskDispatched} = require("firebase-functions/tasks");
exports.myTask = onTaskDispatched({retryConfig: {...}}, (req) => {/*...*/});

// Custom Events
const {onCustomEventPublished} = require("firebase-functions/eventarc");
exports.myHandler = onCustomEventPublished("event.type", (event) => {/*...*/});
```

### Python 2nd Gen

```python
from firebase_functions import firestore_fn, database_fn, https_fn, pubsub_fn, tasks_fn

# Firestore
@firestore_fn.on_document_created(document="messages/{id}")
def my_trigger(event: firestore_fn.Event):
    pass

# Realtime Database
@database_fn.on_value_created(reference="/messages/{id}")
def my_trigger(event: database_fn.Event):
    pass

# HTTPS
@https_fn.on_request()
def my_endpoint(req: https_fn.Request) -> https_fn.Response:
    pass

# Callable
@https_fn.on_call()
def my_callable(req: https_fn.CallableRequest):
    pass

# Pub/Sub
@pubsub_fn.on_message_published(topic="my-topic")
def my_handler(event: pubsub_fn.CloudEvent):
    pass

# Task Queue
@tasks_fn.on_task_dispatched(retryConfig=tasks_fn.RetryConfig(...))
def my_task(req: tasks_fn.Request):
    pass
```

---

## Common Patterns

### Pattern 1: Data Transformation
```
Event → Read → Transform → Write Back
```
Example: Uppercase text, compute aggregates, denormalize data

### Pattern 2: Security Gate
```
Auth Event → Validate → Accept or Reject
```
Example: Domain restrictions, banned user checks

### Pattern 3: Client-Triggered Action
```
Client Callable → Validate → Process → Return Result
```
Example: Add to cart, submit form, generate report

### Pattern 4: Asynchronous Integration
```
Event → Enqueue Task → External API → Store Result
```
Example: Send emails, post to Slack, backup data

### Pattern 5: File Processing Pipeline
```
Upload → Detect → Process → Store Variants
```
Example: Image thumbnails, video transcoding, document OCR

### Pattern 6: Real-Time Notification
```
Database Change → Lookup Recipients → Send FCM
```
Example: Follow notifications, chat messages, alerts

---

## Admin SDK Quick Reference

### Firestore

```javascript
const {getFirestore} = require("firebase-admin/firestore");
const db = getFirestore();

// Write
await db.collection("users").doc(uid).set({name: "Alice"});
await db.collection("users").doc(uid).update({lastSeen: new Date()});
await db.collection("messages").add({text: "Hello"});

// Read
const doc = await db.collection("users").doc(uid).get();
const data = doc.data();

// Query
const snapshot = await db.collection("users").where("age", ">", 18).get();
snapshot.forEach(doc => console.log(doc.data()));

// Delete
await db.collection("users").doc(uid).delete();
```

### Realtime Database

```javascript
const {getDatabase} = require("firebase-admin/database");
const db = getDatabase();

// Write
await db.ref("/users/user1").set({name: "Alice"});
await db.ref("/users/user1/score").set(100);

// Read
const snapshot = await db.ref("/users/user1").once("value");
const data = snapshot.val();

// Delete
await db.ref("/users/user1").remove();
```

### Cloud Storage

```javascript
const {getStorage} = require("firebase-admin/storage");
const bucket = getStorage().bucket();

// Download
const [buffer] = await bucket.file("images/photo.jpg").download();

// Upload
await bucket.file("images/photo.jpg").save(buffer, {
  metadata: {contentType: "image/jpeg"},
});

// Delete
await bucket.file("images/photo.jpg").delete();
```

### Authentication

```javascript
const {getAuth} = require("firebase-admin/auth");

// Get user
const user = await getAuth().getUser(uid);
const userByEmail = await getAuth().getUserByEmail("alice@example.com");

// Create user
const newUser = await getAuth().createUser({
  email: "alice@example.com",
  password: "secretPassword",
});

// Delete user
await getAuth().deleteUser(uid);

// Custom token
const token = await getAuth().createCustomToken(uid);
```

### Cloud Messaging

```javascript
const {getMessaging} = require("firebase-admin/messaging");

// Send to single device
await getMessaging().send({
  token: deviceToken,
  notification: {title: "Hello", body: "World"},
  data: {key: "value"},
});

// Send to multiple devices
const messages = tokens.map(token => ({token, notification: {...}}));
const response = await getMessaging().sendEach(messages);
console.log(`${response.successCount} sent, ${response.failureCount} failed`);
```

---

## Error Handling Best Practices

### Callable Functions

```javascript
const {HttpsError} = require("firebase-functions/https");

exports.myCallable = onCall((request) => {
  // Validation
  if (!request.data.email) {
    throw new HttpsError("invalid-argument", "Email required");
  }

  // Authentication
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Must be logged in");
  }

  // Authorization
  if (!hasPermission(request.auth.uid)) {
    throw new HttpsError("permission-denied", "No access");
  }

  // Server errors
  try {
    return doWork();
  } catch (error) {
    throw new HttpsError("internal", error.message);
  }
});
```

### Event Triggers

```javascript
exports.myTrigger = onDocumentCreated("/path/{id}", async (event) => {
  try {
    await processDocument(event.data);
  } catch (error) {
    // Log error
    console.error("Failed to process:", error);

    // Re-throw to trigger retry (with exponential backoff)
    throw error;
  }
});
```

**Error Codes**:
- `invalid-argument` - Bad input
- `unauthenticated` - Not logged in
- `permission-denied` - Insufficient permissions
- `not-found` - Resource doesn't exist
- `already-exists` - Duplicate resource
- `resource-exhausted` - Rate limit exceeded
- `failed-precondition` - Precondition not met
- `internal` - Server error
- `unavailable` - Service unavailable

---

## Configuration & Secrets

### Environment Variables (Public Config)

```bash
# Set
firebase functions:config:set stripe.key="pk_test_123" stripe.secret="sk_test_456"

# Get
firebase functions:config:get

# Use in code
const config = functions.config();
const apiKey = config.stripe.key;
```

### Secret Manager (Sensitive Data)

```bash
# Set secret
firebase functions:secrets:set STRIPE_SECRET_KEY
# Enter value when prompted

# Use in code
const stripeKey = process.env.STRIPE_SECRET_KEY;
```

### Local Development (.env file)

```env
# .env file
DATABASE_URL=https://my-project.firebaseio.com
API_KEY=test-api-key
```

```javascript
// Load in code
require("dotenv").config();
const apiKey = process.env.API_KEY;
```

---

## Testing Strategies

### Unit Testing (Jest)

```javascript
// functions/index.js
exports.addnumbers = onCall((request) => {
  return {result: request.data.first + request.data.second};
});

// tests/index.test.js
const {addnumbers} = require("../index");

describe("addnumbers", () => {
  it("adds two numbers", () => {
    const result = addnumbers({
      data: {first: 2, second: 3},
      auth: {uid: "test-user"},
    });
    expect(result.result).toBe(5);
  });
});
```

### Integration Testing

```bash
# Start emulator
firebase emulators:start

# Run tests against emulator
npm test
```

### Load Testing

Use tools like Apache Bench or Locust to test production performance:

```bash
ab -n 1000 -c 10 https://us-central1-myproject.cloudfunctions.net/myFunction
```

---

## Deployment Best Practices

### Development Workflow

```bash
# 1. Local development
firebase emulators:start

# 2. Test
npm test

# 3. Deploy to staging (use project alias)
firebase use staging
firebase deploy --only functions

# 4. Test in staging
# Run smoke tests

# 5. Deploy to production
firebase use production
firebase deploy --only functions
```

### Resource Configuration

```javascript
// Allocate more resources for heavy processing
exports.processVideo = onObjectFinalized({
  timeoutSeconds: 540,  // 9 minutes (max)
  memory: "2GiB",       // More memory
  cpu: 2,               // More CPU cores
}, async (event) => {
  // Heavy video processing
});
```

### Regional Deployment

```javascript
// Deploy to multiple regions
exports.myFunction = onRequest({
  region: ["us-west1", "us-east1", "europe-west1"],
}, (req, res) => {
  res.send("Hello from nearest region");
});
```

---

## Common Pitfalls

### 1. Missing Return Statements
```javascript
// BAD - Promise not returned
exports.badExample = onDocumentCreated("/path/{id}", (event) => {
  doAsyncWork();  // Promise not returned!
});

// GOOD
exports.goodExample = onDocumentCreated("/path/{id}", async (event) => {
  await doAsyncWork();  // Awaited
});
```

### 2. Infinite Loops
```javascript
// BAD - Thumbnail trigger will fire on thumbnails
exports.badThumbnail = onObjectFinalized((event) => {
  createThumbnail(event.data.name);  // Creates thumb_file.jpg, triggers again!
});

// GOOD - Filter out thumbnails
exports.goodThumbnail = onObjectFinalized((event) => {
  if (event.data.name.startsWith("thumb_")) return;  // Skip thumbnails
  createThumbnail(event.data.name);
});
```

### 3. Unhandled Errors
```javascript
// BAD - Error swallowed
exports.badExample = onCall(async (request) => {
  try {
    return await riskyOperation();
  } catch (error) {
    console.log(error);  // Logged but not communicated to client!
  }
});

// GOOD - Error propagated
exports.goodExample = onCall(async (request) => {
  try {
    return await riskyOperation();
  } catch (error) {
    throw new HttpsError("internal", error.message);
  }
});
```

### 4. Cold Start Delays
```javascript
// BAD - Heavy imports inside function
exports.badExample = onRequest((req, res) => {
  const heavyLibrary = require("heavy-library");  // Loaded on every invocation!
  res.send(heavyLibrary.doWork());
});

// GOOD - Import at top level
const heavyLibrary = require("heavy-library");
exports.goodExample = onRequest((req, res) => {
  res.send(heavyLibrary.doWork());
});
```

### 5. Timeout Issues
```javascript
// BAD - Long synchronous loop
exports.badExample = onDocumentCreated("/items/{id}", async (event) => {
  for (let i = 0; i < 10000; i++) {
    await processItem(i);  // Serial processing - very slow!
  }
});

// GOOD - Use Task Queue for batch work
exports.goodExample = onDocumentCreated("/items/{id}", async (event) => {
  const queue = getFunctions().taskQueue("processItems");
  const tasks = items.map(item => ({itemId: item.id}));
  await queue.enqueue(tasks);  // Distributed processing
});
```

---

## Production Observability

### Logging

```javascript
const {logger} = require("firebase-functions");

exports.myFunction = onRequest((req, res) => {
  logger.log("Info message", {user: req.body.userId});
  logger.info("Informational log");
  logger.warn("Warning message");
  logger.error("Error occurred", {error: err.message});
  logger.debug("Debug info");

  res.send("OK");
});
```

View logs:
```bash
firebase functions:log
firebase functions:log --only myFunction
```

### OpenTelemetry (Distributed Tracing)

```javascript
const {trace} = require("@opentelemetry/api");

exports.complexOperation = onRequest(async (req, res) => {
  const tracer = trace.getTracer("my-functions");

  const span = tracer.startSpan("process-request");
  try {
    // Your work here
    const result = await doWork();
    span.setAttributes({resultCount: result.length});
    res.json(result);
  } catch (error) {
    span.recordException(error);
    throw error;
  } finally {
    span.end();
  }
});
```

### Monitoring

- **Firebase Console**: View invocation counts, error rates, execution times
- **Cloud Monitoring**: Set up alerts for error rates, latency, timeouts
- **Cloud Trace**: View distributed traces across services

---

## Security Checklist

- [ ] **Validate all inputs** in callable functions
- [ ] **Check authentication** before sensitive operations
- [ ] **Use HttpsError** for user-facing errors
- [ ] **Store secrets in Secret Manager**, not code
- [ ] **Use IAM roles** for service account permissions
- [ ] **Enable CORS** only for trusted origins
- [ ] **Rate limit** public endpoints
- [ ] **Sanitize user input** before database writes
- [ ] **Audit logs** for sensitive operations
- [ ] **Use HTTPS-only** endpoints

---

## Performance Optimization

### 1. Connection Pooling

```javascript
// BAD - New connection per invocation
exports.badExample = onRequest(async (req, res) => {
  const db = await connectToDatabase();  // Slow!
  res.json(await db.query());
});

// GOOD - Reuse connection
let dbConnection = null;
exports.goodExample = onRequest(async (req, res) => {
  if (!dbConnection) dbConnection = await connectToDatabase();
  res.json(await dbConnection.query());
});
```

### 2. Minimize Dependencies

Only install what you need. Each dependency increases cold start time.

```bash
# Check bundle size
npm ls --depth=0
```

### 3. Use Appropriate Memory Allocation

```javascript
// Light function - use default (256 MB)
exports.lightFunction = onRequest((req, res) => {/*...*/});

// Heavy function - allocate more
exports.heavyFunction = onRequest({memory: "2GiB"}, (req, res) => {/*...*/});
```

### 4. Regional Deployment

Deploy close to users:

```javascript
exports.usFunction = onRequest({region: "us-central1"}, handler);
exports.euFunction = onRequest({region: "europe-west1"}, handler);
```

---

## Integration Examples

### Slack Webhook

```javascript
exports.postToSlack = onDocumentCreated("/alerts/{id}", async (event) => {
  const alert = event.data.data();

  await fetch(process.env.SLACK_WEBHOOK_URL, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      text: `Alert: ${alert.message}`,
      channel: "#alerts",
    }),
  });
});
```

### SendGrid Email

```javascript
const sgMail = require("@sendgrid/mail");
sgMail.setApiKey(process.env.SENDGRID_API_KEY);

exports.sendWelcomeEmail = onDocumentCreated("/users/{uid}", async (event) => {
  const user = event.data.data();

  await sgMail.send({
    to: user.email,
    from: "noreply@myapp.com",
    subject: "Welcome!",
    text: `Welcome ${user.name}!`,
  });
});
```

### Stripe Payment

```javascript
const stripe = require("stripe")(process.env.STRIPE_SECRET_KEY);

exports.createPayment = onCall(async (request) => {
  if (!request.auth) throw new HttpsError("unauthenticated");

  const paymentIntent = await stripe.paymentIntents.create({
    amount: request.data.amount,
    currency: "usd",
    metadata: {userId: request.auth.uid},
  });

  return {clientSecret: paymentIntent.client_secret};
});
```

---

## Reference: All 70+ Examples

### Quickstarts (13 examples)
- uppercase-firestore, uppercase-rtdb
- https-time-server, callable-functions, callable-functions-streaming
- thumbnails, pubsub-helloworld
- auth-blocking-functions, firestore-sync-auth
- taskqueues-backup-images, delete-unused-accounts-cron
- alerts-to-discord, custom-events, monitor-cloud-logging, testlab-matrix-completed

### Image Processing (4 examples - 1st gen)
- image-maker, convert-images, moderate-images, exif-images

### Data Consistency (4 examples - 1st gen)
- lastmodified-tracking, child-count, limit-children, delete-old-child-nodes

### Messaging (2 examples)
- fcm-notifications

### Testing (3 examples)
- test-functions-jest, test-functions-jest-ts, test-functions-mocha

### Observability (1 example)
- instrument-with-opentelemetry

### Integrations (30+ examples - 1st gen)
- Slack, Discord, GitHub, Stripe, PayPal
- BigQuery, Google Sheets, YouTube
- OAuth providers (Okta, LinkedIn, Spotify, Instagram)
- Text translation, moderation, search
- And many more...

---

## Next Steps for AI Agents

1. **Start with Quickstarts**: Work through weeks 1-8 sequentially
2. **Build a Project**: Choose a use case and implement end-to-end
3. **Study Integration Examples**: Learn external service patterns
4. **Test Everything**: Write unit and integration tests
5. **Deploy to Production**: Use staging → production workflow
6. **Monitor and Optimize**: Use logs, traces, and metrics

**Key Resources**:
- Repository: https://github.com/firebase/functions-samples
- Firebase Docs: https://firebase.google.com/docs/functions
- Node.js Reference: https://firebase.google.com/docs/reference/functions
- Python Reference: https://firebase.google.com/docs/reference/functions/python

---

## Summary

Firebase Cloud Functions enable serverless, event-driven application logic. Key takeaways:

1. **Trigger Types**: Event-driven (Firestore, RTDB, Storage, Pub/Sub, Auth) vs Request-response (HTTPS, Callable)
2. **Admin SDK**: Interact with all Firebase services from functions
3. **Patterns**: Data transformation, security gates, client actions, integrations, file processing, notifications
4. **Testing**: Jest, Mocha, emulator-based testing
5. **Production**: Error handling, logging, monitoring, secrets management
6. **Performance**: Connection pooling, dependency minimization, regional deployment

**70+ official examples** cover every pattern you'll need. Start simple, iterate, and build production-grade serverless applications.
