// /**
//  * i18n – đa ngôn ngữ với i18next + react-i18next.
//  * Ngôn ngữ mặc định theo thiết bị (expo-localization), fallback: vi.
//  */
// import i18n from "i18next";
// import { initReactI18next } from "react-i18next";
// import * as Localization from "expo-localization";

// import en from "../locales/en.json";
// import vi from "../locales/vi.json";

// // resources: khai báo danh sách ngôn ngữ + file dịch tương ứng
// // i18next sẽ đọc key trong "translation"
// const resources = {
//   en: { translation: en },
//   vi: { translation: vi },
// };

// // deviceLocale: lấy languageCode của thiết bị (vd: "en", "vi")
// // nếu không lấy được thì mặc định "vi"
// const deviceLocale = Localization.getLocales()[0]?.languageCode ?? "vi";

// // supportedCodes: chỉ cho phép các ngôn ngữ app hỗ trợ
// const supportedCodes = ["en", "vi"];

// // lang: nếu ngôn ngữ thiết bị nằm trong supportedCodes thì dùng, không thì fallback "vi"
// const lang = supportedCodes.includes(deviceLocale) ? deviceLocale : "vi";

// // init i18next + bridge cho react-i18next (hook/useTranslation)
// i18n.use(initReactI18next).init({
//   resources,              // nguồn bản dịch
//   lng: lang,              // ngôn ngữ hiện tại
//   fallbackLng: "vi",      // fallback nếu thiếu key hoặc ngôn ngữ không hỗ trợ
//   compatibilityJSON: "v4",// tương thích format JSON (thường cần cho RN/Expo)
//   interpolation: {
//     escapeValue: false,   // RN không render HTML nên không cần escape
//   },
// });

// export default i18n;
