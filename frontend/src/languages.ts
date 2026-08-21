export interface LanguageOption {
  code: string;
  label: string;
  nativeLabel: string;
  placeholder: string;
  prompts: string[];
}

export const LANGUAGES: LanguageOption[] = [
  {
    code: "hi",
    label: "Hindi",
    nativeLabel: "हिन्दी",
    placeholder: "Ask in Hindi…",
    prompts: [
      "भारत का राष्ट्रीय खेल क्या है?",
      "इंटरनेट कैसे काम करता है?",
      "फ़ायरवॉल का उद्देश्य क्या है?",
      "भारत की राजधानी क्या है?",
    ],
  },
  {
    code: "bn",
    label: "Bengali",
    nativeLabel: "বাংলা",
    placeholder: "Ask in Bengali…",
    prompts: [
      "ভারতের জাতীয় খেলা কী?",
      "ইন্টারনেট কীভাবে কাজ করে?",
      "ফায়ারওয়ালের উদ্দেশ্য কী?",
      "ভারতের রাজধানী কী?",
    ],
  },
  {
    code: "en",
    label: "English",
    nativeLabel: "English",
    placeholder: "Ask in English…",
    prompts: [
      "What is India's national sport?",
      "How does the internet work?",
      "What is the purpose of a firewall?",
      "What is the capital of India?",
    ],
  },
];

export const DEFAULT_LANGUAGE_CODE = LANGUAGES[0].code;

export const languageByCode = (code: string): LanguageOption =>
  LANGUAGES.find((option) => option.code === code) ?? LANGUAGES[0];

/** "हिन्दी · Hindi", but just "English" when there's no separate native form. */
export const languageDisplayName = (option: LanguageOption): string =>
  option.nativeLabel === option.label ? option.label : `${option.nativeLabel} · ${option.label}`;
