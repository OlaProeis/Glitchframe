# Update Handover Instructions

Task is complete. Update the handover for the next session.

task 54 is next

---

## 1. Mark Current Task Done
Use Task Master MCP tool to set status:
`set_task_status --id=<current-task-id> --status=done`
Prefer MCP tools over CLI commands for Task Master operations.

## 2. Create Documentation
Create feature-based documentation for completed work.
1. Identify what was implemented (group by feature, not task).
2. Create doc in `docs/` or `docs/technical/`.
3. **Update `docs/index.md`** with the new entry and a 1-line description.
*Naming:* Good: `github-api-client.md`. Bad: `task-1.md`.

## 3. Get Next Task
Fetch the next task using: `next_task` or `get_task --id=<next-task-id>`

## 4. Update current-handover-prompt.md
Replace the current task sections with the new task:
- **Current Task:** Full details of next task (ID, title, description, complexity, etc.)
- **Key Files:** Only files relevant to the NEW task.
- **Context:** Only if needed for the new task.
*Crucial Rule:* Remove ANY previous task details. Do NOT accumulate a task history.

## 5. Update ai-context.md (if needed)
If the completed task changed the architecture significantly:
- Update the Architecture section or "Where Things Live".
- Keep it under ~100 lines.

## 6. Verification Checklist
- [ ] Current task marked as `done` in Task Master
- [ ] Feature documentation created
- [ ] `docs/index.md` updated with new doc entry
- [ ] `current-handover-prompt.md` updated with ONLY next task context
- [ ] Code compiles and tests pass clean
