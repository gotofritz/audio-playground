# REQUIREMENTS

- I want to make the code more modular, with modules shared across commands. For example a Segmenter module, etc
- For large packages like torch or sam-audio, I would like to import only when actually needed
- extract sam-audio will stay, and in the future there will be a `extract demucs and maybe more`
- extract sam-audio currently works in a few steps: convert input to wav if needed; split into segments; run each segments through the model; concatenate the output segments. I would like those steps to be runnable standalone AND as part of the extract sam-audio command (and potentially other); for example "convert to wav if needed" or "segment" could be standalone commands. There is already a --continue-from which kind of does that, as it starts from the 'concatenate output segments' step
- I would like to allow users to run sequence of commands via a yaml file, where the top level items are - metadata: {} and - commands: []. Each command would be a call to an `audio-playground command, with args. That would be run with a new command, `audio-playground run --config xxxx``
- When a command is run, all args are passed either as CLI arg, or read from the AudioPlaygroundConfig object; that means each command should be able to override every attribute of AudioPlaygroundConfig from the cli (be smart about this, share those and do not repeat them in each command)
- I would like to add args to the extract command to define an input window, so that it won't apply to the whole file
- High test coverage >90%, but only as a last step when we have fixed everything else
