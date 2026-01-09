StudyBuddy v4 is the first "real" version of the app in terms of the data being related to AI Engineering. The frontend needs to mirror that and become the first "real" version of the interface.

To that end, let's brainstorm some UI ideas. Here are a few requirements and ideas to ponder:
- This is not a chat-first app. There needs to be chat, but it is secondary.
- The main purpose of StudyBuddy is to help the user learn using spaced-repetition with flashcards.
- The data covers the entire book, but the user will only need that data incrementally. For example, they might come to StudyBuddy after Chapter 4 and want to reinforce what they learned in that chapter. There needs to be some way to specify the learning context (e.g., just Chapter 4 or all of Chapters 1 through 4, etc.). I have in mind a dropdown with the chapter titles. The user picks a chapter, then somehow specifies whether they want to study just that chapter or all the chapters up to and including the selected chapter.
- The book's TOC is added to the data, so when the user selects a chapter, the AI knows generally what topics the user wants to study (which, again, might be just that chapter or that chapter and all of the preceding chapters).
- The StudyBuddy I'll build will focus on AI Engineering by loading the `documents` folder with the AI Engineering reference guides (the files in this project named `ref-*.md`) and the book's TOC. However, StudyBuddy should be flexible enough to enable the user to study anything by loading `documents` with their study data (such as a textbook or a collection of documents) and some equivalent of a TOC that specifies the topics and subtopics they want to study. So, instead of a TOC, we need to come up with some kind of `topic-list.md` file format, probably just a collection of headings for the main topics and bulleted lists for the subtopics.
- It's important that the app be mobile-first, because I can imagine users firing up the app to study while waiting in line, at the dentist's office, etc.

I don't need any code yet. Let's start with your reactions to the above ideas and go from there.


Frontend requirements:
- Vanilla HTML, CSS, and JavaScript only (no frameworks)
- Mobile-first design
- Fully accessible for all users
- Clean, minimal design
- Full-height layout with messages scrolling in the middle
- Text input fixed at the bottom with a send button
- Messages should:
  - Show user messages right-aligned with a different background color
  - Show AI responses left-aligned
  - Auto-scroll to newest message
  - Display a typing indicator while waiting for AI response
- Header at top with app name "StudyBuddy"
- Handle errors gracefully (show error message if backend is down)
- Keyboard shortcut: Enter to send (Shift+Enter for new line)

Style preferences:
- Modern, professional look
- Good use of whitespace
- Readable font sizes
- Subtle shadows/borders, nothing too flashy
- Use a pleasant color scheme (blues/grays work great)