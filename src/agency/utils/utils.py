def print_logs(log):
    print("--------\n")
    for msg in log:
        print(msg)
    print("--------\n")


def generate_messages(conversation_history, student_prompt, system_msg):

    messages = [{"role": "system", "content": system_msg}]
    messages.extend([
            {"role": m["role"], "content": m["content"]}
            for m in conversation_history
        ])
    messages.append({"role": "user", "content": student_prompt})

    return messages